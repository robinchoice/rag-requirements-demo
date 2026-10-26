import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    lang         text NOT NULL CHECK (lang IN ('de', 'en')),
    path         text NOT NULL,
    unit         text NOT NULL,
    section      text,
    heading      text NOT NULL,
    text         text NOT NULL,
    position     int NOT NULL,
    content_hash text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (
        to_tsvector(
            CASE lang WHEN 'de' THEN 'german'::regconfig ELSE 'english'::regconfig END,
            heading || ' ' || text
        )
    ) STORED,
    UNIQUE (lang, path)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_unit_idx ON chunks (lang, unit, position);
"""


def migrate(conn: psycopg.Connection, dim: int) -> None:
    conn.execute(SCHEMA.format(dim=int(dim)))


def configure(conn: psycopg.Connection) -> None:
    # HNSW applies the WHERE filter after the index scan; iterative scans keep walking
    # the graph until LIMIT rows survive it (pgvector >= 0.8).
    conn.execute("SET hnsw.iterative_scan = strict_order")


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
    configure(conn)
    return conn


def pool(url: str) -> ConnectionPool:
    return ConnectionPool(
        url,
        min_size=1,
        max_size=5,
        open=True,
        configure=configure,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"
