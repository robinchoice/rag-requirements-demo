"""Fetch the MDR, chunk it and load chunks with embeddings into Postgres.

Idempotent: raw files are downloaded once, unchanged chunks are not re-embedded,
chunks that no longer exist are deleted.

    uv run --extra ingest python -m mdr_rag.ingest
"""

import argparse
import hashlib
from pathlib import Path

import httpx
import psycopg

from mdr_rag import db
from mdr_rag.chunking import Chunk, blocks_of, chunk_blocks
from mdr_rag.config import load_settings
from mdr_rag.llm import Llm, OpenAiLlm

# Cellar REST API of the EU Publications Office: the machine interface behind EUR-Lex.
SOURCE_URL = "http://publications.europa.eu/resource/celex/32017R0745"
CELLAR_LANGS = {"de": "deu", "en": "eng"}


def fetch(lang: str, raw_dir: Path) -> Path:
    path = raw_dir / f"mdr_{lang}.html"
    if not path.exists():
        response = httpx.get(
            SOURCE_URL,
            headers={"Accept": "application/xhtml+xml", "Accept-Language": CELLAR_LANGS[lang]},
            follow_redirects=True,
            timeout=120,
        )
        response.raise_for_status()
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def parse(path: Path, lang: str) -> list[Chunk]:
    from docling.document_converter import DocumentConverter  # heavy, only needed here

    document = DocumentConverter().convert(path).document
    return chunk_blocks(blocks_of(document), lang)


def content_hash(chunk: Chunk, embed_model: str) -> str:
    return hashlib.sha256(f"{embed_model}\n{chunk.embedding_text}".encode()).hexdigest()


UPSERT = """
INSERT INTO chunks (lang, path, unit, section, heading, text, position, content_hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (lang, path) DO UPDATE SET
    unit = excluded.unit, section = excluded.section, heading = excluded.heading,
    text = excluded.text, content_hash = excluded.content_hash, embedding = excluded.embedding
"""
BATCH = 16


def load(conn: psycopg.Connection, llm: Llm, embed_model: str, chunks: list[Chunk]) -> dict:
    """Upsert chunks of one language; returns counts of embedded, unchanged and deleted.

    Embeddings are committed batch by batch, so an interrupted run resumes where it
    stopped (embedding is the slow part on a CPU).
    """
    lang = chunks[0].lang
    existing = {
        row["path"]: row["content_hash"]
        for row in conn.execute("SELECT path, content_hash FROM chunks WHERE lang = %s", (lang,))
    }
    hashes = {c.path: content_hash(c, embed_model) for c in chunks}
    changed = [c for c in chunks if existing.get(c.path) != hashes[c.path]]

    for i in range(0, len(changed), BATCH):
        batch = changed[i : i + BATCH]
        vectors = llm.embed([c.embedding_text for c in batch])
        with conn.transaction():
            for c, vector in zip(batch, vectors, strict=True):
                conn.execute(
                    UPSERT,
                    (c.lang, c.path, c.unit, c.section, c.heading, c.text, c.position,
                     hashes[c.path], db.vector_literal(vector)),
                )  # fmt: skip
        print(f"{lang}: embedded {i + len(batch)}/{len(changed)}", flush=True)

    with conn.transaction():
        for c in chunks:  # positions may shift without the content changing
            conn.execute(
                "UPDATE chunks SET position = %s WHERE lang = %s AND path = %s",
                (c.position, lang, c.path),
            )
        stale = [p for p in existing if p not in hashes]
        conn.execute("DELETE FROM chunks WHERE lang = %s AND path = ANY(%s)", (lang, stale))

    return {
        "embedded": len(changed),
        "unchanged": len(chunks) - len(changed),
        "deleted": len(stale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", nargs="+", default=["de", "en"], choices=["de", "en"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    settings = load_settings()
    llm = OpenAiLlm(settings)
    with db.connect(settings.database_url) as conn:
        db.migrate(conn, settings.embed_dim)
        for lang in args.lang:
            chunks = parse(fetch(lang, args.raw_dir), lang)
            counts = load(conn, llm, settings.embed_model, chunks)
            print(f"{lang}: {len(chunks)} chunks, {counts}")


if __name__ == "__main__":
    main()
