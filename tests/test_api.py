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
    with TestClient(create_app(llm=llm, chat_model=object(), database_url=database_url)) as client:
        client.llm = llm
        yield client


def test_search(client):
    response = client.get("/search", params={"q": "Pflichten der Importeure", "k": 2})

    assert response.status_code == 200
    assert [h["path"] for h in response.json()][0] == ART_13


def test_ask_keeps_only_citations_of_retrieved_passages(client):
    client.llm.replies = [
        "Hersteller brauchen ein QM-System [1]. Das gilt immer [9]."
        " Daraus folgt eine Dokumentationspflicht (gefolgert)."
    ]

    body = client.post(
        "/ask", json={"question": "Was müssen Hersteller beim QM-System tun?"}
    ).json()
    prompt = client.llm.prompts[0][1]["content"]

    assert body["found"] is True
    assert body["lang"] == "de"
    assert body["answer"] == (
        f"Hersteller brauchen ein QM-System [{ART_10}]. Das gilt immer."
        " Daraus folgt eine Dokumentationspflicht (gefolgert)."
    )
    assert [c["path"] for c in body["citations"]] == [ART_10]
    assert body["citations"][0]["url"].endswith("CELEX:32017R0745#art_10")
    assert prompt.startswith("[1] Artikel 10 – Pflichten der Hersteller\n")


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
                     "addressee": "manufacturer", "excerpt": 1},
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
    assert body["truncated"] is False


def test_extract_requirements_validates_input(client):
    assert client.post("/extract-requirements", json={}).status_code == 422
    assert client.post("/extract-requirements", json={"article": "§ 5"}).status_code == 422
    assert client.post("/extract-requirements", json={"article": "Art. 99"}).status_code == 404


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok", "chunks": 4}
    client.post("/ask", json={"question": "Wie backe ich ein Sauerteigbrot?"})

    metrics = client.get("/metrics").text

    assert 'mdr_ask_total{found="false"}' in metrics
    assert "http_request_duration_seconds" in metrics


def test_extraction_schema_only_allows_excerpts_that_were_sent():
    schema = answer.extraction_schema(3).model_json_schema()

    assert schema["$defs"]["ExtractedRequirement"]["properties"]["excerpt"]["enum"] == [1, 2, 3]
