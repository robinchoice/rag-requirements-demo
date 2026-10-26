"""Hybrid retrieval without a framework: vector (HNSW) + full text, fused by RRF.

The cascade follows "cheap, high-recall stages first, precise stage last":
both retrievers return generous candidate lists (recall), Reciprocal Rank Fusion
merges them by rank only (no score calibration needed), and the expensive precise
step - the LLM reading whole passages - runs on the few survivors.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import psycopg

from mdr_rag.db import vector_literal
from mdr_rag.llm import Llm

Lang = Literal["de", "en"]
Mode = Literal["hybrid", "vector", "fulltext"]

TS_CONFIG = {"de": "german", "en": "english"}
CANDIDATES = 50
RRF_K = 60

FIELDS = ["id", "lang", "path", "unit", "section", "heading", "text"]
COLUMNS = ", ".join(FIELDS)


@dataclass
class Hit:
    id: int
    lang: str
    path: str
    unit: str
    section: str | None
    heading: str
    text: str
    score: float = 0.0
    # rank (1-based) per retriever, e.g. {"vector": 3, "fulltext": 1}
    ranks: dict[str, int] = field(default_factory=dict)
    similarity: float | None = None


def vector_search(
    conn: psycopg.Connection, embedding: list[float], lang: Lang, k: int = CANDIDATES
) -> list[Hit]:
    rows = conn.execute(
        f"""
        SELECT {COLUMNS}, 1 - (embedding <=> %(q)s::vector) AS similarity
        FROM chunks WHERE lang = %(lang)s
        ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s
        """,
        {"q": vector_literal(embedding), "lang": lang, "k": k},
    ).fetchall()
    return [
        _hit(r, score=r["similarity"], ranks={"vector": i}, similarity=r["similarity"])
        for i, r in enumerate(rows, start=1)
    ]


def fulltext_search(
    conn: psycopg.Connection, query: str, lang: Lang, k: int = CANDIDATES
) -> list[Hit]:
    # A question ANDed term by term rarely matches anything, so the query is the OR
    # of its stemmed, stopword-free lexemes; ts_rank_cd rewards dense co-occurrence.
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
            FROM unnest(tsvector_to_array(to_tsvector(%(cfg)s::regconfig, %(text)s))) AS lexeme
        )
        SELECT {COLUMNS}, ts_rank_cd(tsv, q.query) AS rank
        FROM chunks, q
        WHERE lang = %(lang)s AND tsv @@ q.query
        ORDER BY rank DESC, id LIMIT %(k)s
        """,
        {"cfg": TS_CONFIG[lang], "text": query, "lang": lang, "k": k},
    ).fetchall()
    return [_hit(r, score=r["rank"], ranks={"fulltext": i}) for i, r in enumerate(rows, start=1)]


def _hit(row: dict, **extra) -> Hit:
    return Hit(**{f: row[f] for f in FIELDS}, **extra)


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """score(d) = sum over rankings of 1 / (k + rank of d)."""
    fused: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            merged = fused.setdefault(hit.id, Hit(**{**hit.__dict__, "score": 0.0, "ranks": {}}))
            merged.score += 1 / (k + rank)
            merged.ranks.update(hit.ranks)
            if hit.similarity is not None:
                merged.similarity = hit.similarity
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


def search(
    conn: psycopg.Connection,
    llm: Llm,
    query: str,
    lang: Lang,
    k: int = 10,
    mode: Mode = "hybrid",
) -> list[Hit]:
    rankings = []
    if mode in ("hybrid", "vector"):
        rankings.append(vector_search(conn, llm.embed([query])[0], lang))
    if mode in ("hybrid", "fulltext"):
        rankings.append(fulltext_search(conn, query, lang))
    if mode != "hybrid":
        return rankings[0][:k]
    return reciprocal_rank_fusion(rankings)[:k]


GERMAN_HINTS = re.compile(
    r"\b(der|die|das|und|ist|sind|welche|wer|wie|was|muss|müssen|für|einen?|nicht|nach)\b"
    r"|[äöüß]",
    re.IGNORECASE,
)
ENGLISH_HINTS = re.compile(
    r"\b(the|and|is|are|which|who|how|what|must|shall|for|an?|not|does|do)\b", re.IGNORECASE
)


def detect_lang(text: str) -> Lang:
    return "de" if len(GERMAN_HINTS.findall(text)) >= len(ENGLISH_HINTS.findall(text)) else "en"


def unit_chunks(conn: psycopg.Connection, lang: Lang, unit: str) -> list[Hit]:
    """All chunks of an article or annex in document order."""
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM chunks WHERE lang = %s AND unit = %s ORDER BY position",
        (lang, unit),
    ).fetchall()
    return [_hit(r) for r in rows]
