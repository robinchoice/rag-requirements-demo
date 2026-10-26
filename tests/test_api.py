import json

import pytest
from conftest import FakeLlm
from fastapi.testclient import TestClient
from test_retrieval import CHUNKS

from mdr_rag import answer
from mdr_rag.api import create_app
from mdr_rag.ingest import load

ART_10 = "MDR/Kapitel-II/Art-10/Abs-9"
ART_13 = "MDR/Kapitel-II/Art-13/Abs-1"


@pytest.fixture
def client(conn, database_url, monkeypatch):
    load(conn, FakeLlm(), "m", CHUNKS)
    monkeypatch.setattr(answer, "MIN_SIMILARITY", 0.2)  # fake embeddings are coarse
    llm = FakeLlm()
    with TestClient(create_app(llm=llm, database_url=database_url)) as client:
        client.llm = llm
        yield client


def test_search(client):
    response = client.get("/search", params={"q": "Pflichten der Importeure", "k": 2})

    assert response.status_code == 200
    assert [h["path"] for h in response.json()][0] == ART_13


def test_ask_keeps_only_citations_of_retrieved_passages(client):
    client.llm.replies = [
        f"Hersteller brauchen ein QM-System [{ART_10}]. Das gilt immer [MDR/Erfunden/Art-999]."
        " Daraus folgt eine Dokumentationspflicht (gefolgert)."
    ]

    body = client.post(
        "/ask", json={"question": "Was müssen Hersteller beim QM-System tun?"}
    ).json()

    assert body["found"] is True
    assert body["lang"] == "de"
    assert body["answer"] == (
        f"Hersteller brauchen ein QM-System [{ART_10}]. Das gilt immer."
        " Daraus folgt eine Dokumentationspflicht (gefolgert)."
    )
    assert [c["path"] for c in body["citations"]] == [ART_10]
    assert body["citations"][0]["url"].endswith("CELEX:32017R0745#art_10")
    assert ART_10 in client.llm.prompts[0][1]["content"]  # passage was in the context


def test_ask_off_topic_does_not_call_the_model(client):
    body = client.post("/ask", json={"question": "Wie backe ich ein Sauerteigbrot?"}).json()

    assert body == {
        "lang": "de",
        "found": False,
        "answer": "Dazu habe ich in der MDR nichts gefunden.",
        "citations": [],
    }
    assert client.llm.prompts == []


def test_ask_model_says_not_found(client):
    client.llm.replies = ["NOT_FOUND"]

    body = client.post("/ask", json={"question": "Welche Pflichten haben die Hersteller?"}).json()

    assert body["found"] is False


def test_extract_requirements_of_an_article(client):
    client.llm.replies = [
        json.dumps(
            {
                "requirements": [
                    {"statement": "Hersteller unterhalten ein QM-System.",
                     "addressee": "manufacturer", "source": ART_10},
                    {"statement": "Erfunden.", "addressee": "other", "source": "MDR/X"},
                ]
            }
        )
    ]  # fmt: skip

    body = client.post("/extract-requirements", json={"article": "Artikel 10"}).json()

    assert body["requirements"] == [
        {
            "id": "Kapitel-II/Art-10/Abs-9#1",
            "statement": "Hersteller unterhalten ein QM-System.",
            "addressee": "manufacturer",
            "source": ART_10,
            "url": "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32017R0745#art_10",
        }
    ]
    assert body["dropped"] == 1
    assert body["truncated"] is False


def test_extract_requirements_validates_input(client):
    assert client.post("/extract-requirements", json={}).status_code == 422
    assert client.post("/extract-requirements", json={"article": "§ 5"}).status_code == 422
    assert client.post("/extract-requirements", json={"article": "Art. 99"}).status_code == 404
