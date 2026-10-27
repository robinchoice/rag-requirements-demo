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
    # Normalization 2 divides by document length: without it long annex sections,
    # which contain most query words somewhere, outrank the short paragraph that
    # answers (eval: full-text MRR 0.54 -> 0.78).
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
            FROM unnest(tsvector_to_array(to_tsvector(%(cfg)s::regconfig, %(text)s))) AS lexeme
        )
        SELECT {COLUMNS}, ts_rank_cd(tsv, q.query, 2) AS rank
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


ARTICLE_REF = re.compile(r"\b(?i:art\.?|artikels?|article)\s*(\d+)")
ANNEX_REF = re.compile(r"\b(?i:anhang|annex)\s+([IVXL]+)\b")
RULE_REF = re.compile(r"\b(?i:regel|rule)\s+(\d+)\b")


def references(query: str) -> tuple[list[str], list[str]]:
    """Units and classification rules a question names: 'Artikel 120', 'Annex XVI', 'Rule 8'."""
    units = [f"Art-{n}" for n in ARTICLE_REF.findall(query)]
    units += [f"Annex-{n}" for n in ANNEX_REF.findall(query)]
    return units, RULE_REF.findall(query)


def referenced_chunks(
    conn: psycopg.Connection, query: str, lang: Lang, embedding: list[float] | None = None
) -> list[Hit]:
    """Chunks of the articles, annexes and rules (Annex VIII) a question names, most
    similar first. Terse questions like "What does Rule 8 cover?" carry too little
    meaning for the embedding and Postgres full text has no IDF, so a named reference
    is looked up instead of searched for."""
    units, rules = references(query)
    if not units and not rules:
        return []
    # a rule starts with its section number: "6.3. Regel 11", not "gemäß Regel 11"
    rule_heading = rf"\d+\.\d+\.\s+(Regel|Rule)\s+({'|'.join(rules)})\M" if rules else None
    similarity = "1 - (embedding <=> %(q)s::vector)" if embedding else "NULL::float"
    rows = conn.execute(
        f"""
        SELECT {COLUMNS}, {similarity} AS similarity
        FROM chunks
        WHERE lang = %(lang)s
          AND (unit = ANY(%(units)s) OR (unit = 'Annex-VIII' AND text ~ %(rule)s))
        ORDER BY similarity DESC NULLS LAST, unit, position
        """,
        {
            "q": vector_literal(embedding) if embedding else None,
            "lang": lang,
            "units": units,
            "rule": rule_heading,
        },
    ).fetchall()
    hits = [
        _hit(r, ranks={"reference": i}, similarity=r["similarity"])
        for i, r in enumerate(rows, start=1)
    ]
    if rules:
        # a rule can sit deep in a long section: start the passage at the rule, or
        # excerpt limits downstream cut it off
        heading = re.compile(rf"\d+\.\d+\.\s+(?:Regel|Rule)\s+(?:{'|'.join(rules)})\b")
        for hit in hits:
            if hit.unit not in units and (m := heading.search(hit.text)):
                hit.text = hit.text[m.start() :]
    return hits


def search(
    conn: psycopg.Connection,
    llm: Llm,
    query: str,
    lang: Lang,
    k: int = 10,
    mode: Mode = "hybrid",
    resolve_references: bool = True,
) -> list[Hit]:
    """Ranked chunks; with resolve_references the chunks of named articles, annexes
    and rules take up to half of the k places, ahead of the search results."""
    embedding = llm.embed([query])[0] if mode in ("hybrid", "vector") else None
    rankings = []
    if embedding:
        rankings.append(vector_search(conn, embedding, lang))
    if mode in ("hybrid", "fulltext"):
        rankings.append(fulltext_search(conn, query, lang))
    hits = reciprocal_rank_fusion(rankings) if mode == "hybrid" else rankings[0]
    if not resolve_references:
        return hits[:k]
    pinned = referenced_chunks(conn, query, lang, embedding)[: -(-k // 2)]
    ids = {h.id for h in pinned}
    return (pinned + [h for h in hits if h.id not in ids])[:k]


GERMAN_HINTS = re.compile(
    r"\b(der|die|das|und|ist|sind|welche|wer|wie|was|muss|müssen|für|einen?|nicht|nach"
    r"|an|als|zu|von|mit|bei|im|auf)\b"
    r"|[äöüß]",
    re.IGNORECASE,
)
ENGLISH_HINTS = re.compile(
    r"\b(the|and|is|are|which|who|how|what|must|shall|for|a|not|does|do|of|to|with)\b",
    re.IGNORECASE,
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
