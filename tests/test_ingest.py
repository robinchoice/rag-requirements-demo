from dataclasses import replace

from conftest import FakeLlm

from mdr_rag.chunking import Chunk
from mdr_rag.ingest import load


def chunk(path: str, text: str) -> Chunk:
    return Chunk("de", path, "Art-10", path.rsplit("-", 1)[-1], "Artikel 10 – Pflichten", text)


def test_load_is_idempotent(conn):
    chunks = [
        chunk("MDR/Kapitel-II/Art-10/Abs-1", "erste"),
        chunk("MDR/Kapitel-II/Art-10/Abs-2", "zweite"),
    ]
    llm = FakeLlm()

    assert load(conn, llm, "m", chunks) == {"embedded": 2, "unchanged": 0, "deleted": 0}
    assert load(conn, llm, "m", chunks) == {"embedded": 0, "unchanged": 2, "deleted": 0}

    changed = [replace(chunks[0], text="geändert")]
    assert load(conn, llm, "m", changed) == {"embedded": 1, "unchanged": 0, "deleted": 1}
    assert len(llm.embedded) == 3
    rows = conn.execute("SELECT path, text, tsv::text FROM chunks").fetchall()
    assert [(r["path"], r["text"]) for r in rows] == [("MDR/Kapitel-II/Art-10/Abs-1", "geändert")]
    assert "'pflicht'" in rows[0]["tsv"]  # german stemming via the generated tsvector
