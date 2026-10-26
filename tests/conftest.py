import hashlib
import math
import re

import pytest
from testcontainers.community.postgres import PostgresContainer

from mdr_rag import db

DIM = 1024


def fake_embedding(text: str) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words point the same way."""
    vector = [0.0] * DIM
    for word in re.findall(r"\w{3,}", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeLlm:
    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.embedded: list[str] = []
        self.prompts: list[list[dict]] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return [fake_embedding(t) for t in texts]

    def chat(self, messages, schema=None):
        self.prompts.append(messages)
        return self.replies.pop(0)


@pytest.fixture(scope="session")
def database_url():
    with PostgresContainer("pgvector/pgvector:pg17", driver=None) as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def conn(database_url):
    with db.connect(database_url) as connection:
        db.migrate(connection, DIM)
        connection.execute("TRUNCATE chunks")
        yield connection
