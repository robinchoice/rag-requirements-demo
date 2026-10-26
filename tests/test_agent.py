from conftest import FakeLlm
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from test_retrieval import CHUNKS

from mdr_rag import agent, tree
from mdr_rag.agent import run_agent
from mdr_rag.ingest import load


class ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"{name}-{len(args)}"}])


def test_tree_list_and_read(conn):
    load(conn, FakeLlm(), "m", CHUNKS)

    assert tree.list_children(conn, "de", "MDR") == [
        "Kapitel-II/",
        "Anhang-I/  Anhang I – Software",
    ]
    assert tree.list_children(conn, "de", "MDR/Kapitel-II") == [
        "Art-10/  Artikel 10 – Pflichten der Hersteller",
        "Art-13/  Artikel 13 – Pflichten der Importeure",
    ]
    text, paths = tree.read(conn, "de", "MDR/Kapitel-II/Art-10")
    assert paths == ["MDR/Kapitel-II/Art-10/Abs-9"]
    assert text.startswith("Artikel 10 – Pflichten der Hersteller\n\n[MDR/Kapitel-II/Art-10/Abs-9]")
    assert tree.read(conn, "de", "MDR/Kapitel-IX") == ("MDR/Kapitel-IX: no such path", [])


def test_agent_reads_and_cites_only_what_it_has_seen(conn):
    load(conn, FakeLlm(), "m", CHUNKS)
    model = ScriptedChatModel(
        messages=iter(
            [
                call("list", path="MDR/Kapitel-II"),
                call("read", path="MDR/Kapitel-II/Art-10"),
                AIMessage(
                    "Hersteller unterhalten ein QM-System [MDR/Kapitel-II/Art-10]. "
                    "Importeure auch [MDR/Kapitel-II/Art-13/Abs-1] (gefolgert)."
                ),
            ]
        )
    )

    result = run_agent(conn, FakeLlm(), model, "Was müssen Hersteller tun?")

    assert [(s.tool, s.args) for s in result.steps] == [
        ("list", {"path": "MDR/Kapitel-II"}),
        ("read", {"path": "MDR/Kapitel-II/Art-10"}),
    ]
    # Art-13 was only listed, never read or found, so its citation is dropped.
    assert result.answer == (
        "Hersteller unterhalten ein QM-System [MDR/Kapitel-II/Art-10]. Importeure auch (gefolgert)."
    )
    assert [c.path for c in result.citations] == ["MDR/Kapitel-II/Art-10"]


def test_agent_stops_after_max_steps(conn, monkeypatch):
    load(conn, FakeLlm(), "m", CHUNKS)
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    model = ScriptedChatModel(
        messages=iter(
            [
                call("search", query="Hersteller"),
                call("search", query="Hersteller Pflichten"),
                AIMessage("Hersteller brauchen ein QM-System [MDR/Kapitel-II/Art-10/Abs-9]."),
            ]
        )
    )

    result = run_agent(conn, FakeLlm(), model, "Was müssen Hersteller tun?")

    assert len(result.steps) == 2
    assert [c.path for c in result.citations] == ["MDR/Kapitel-II/Art-10/Abs-9"]


def test_agent_endpoint(conn, database_url):
    from fastapi.testclient import TestClient

    from mdr_rag.api import create_app

    load(conn, FakeLlm(), "m", CHUNKS)
    model = ScriptedChatModel(messages=iter([AIMessage("Nichts gelesen [MDR/Kapitel-II/Art-10].")]))
    app = create_app(llm=FakeLlm(), chat_model=model, database_url=database_url)

    with TestClient(app) as client:
        body = client.post("/agent", json={"question": "Was müssen Hersteller tun?"}).json()

    assert body == {"lang": "de", "answer": "Nichts gelesen.", "citations": [], "steps": []}
