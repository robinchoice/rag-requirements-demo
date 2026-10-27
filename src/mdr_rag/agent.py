"""A small LangGraph agent that browses the regulation instead of receiving top-k chunks.

Graph: agent -> (tool calls?) -> tools -> agent ... -> END. An answer before any
read() is sent back once with the request to read the article first (small models
like to answer from search snippets). After MAX_STEPS tool rounds the model gets no
tools any more and must answer with what it has read.
"""

from typing import Annotated, TypedDict

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from mdr_rag import tree
from mdr_rag.answer import INFERRED, LANGUAGE, Citation, citation, verify_citations
from mdr_rag.llm import Llm
from mdr_rag.retrieval import Lang, detect_lang, search

MAX_STEPS = 4
READ_FIRST = "Before answering, read the most relevant article in full with read(path)."


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


class ToolCall(BaseModel):
    tool: str
    args: dict


class AgentAnswer(BaseModel):
    lang: Lang
    answer: str
    citations: list[Citation]
    steps: list[ToolCall]


def system_prompt(lang: Lang) -> str:
    root = "MDR/Kapitel-II/Art-10/Abs-9" if lang == "de" else "MDR/Chapter-II/Art-10/Para-9"
    return (
        "You research the EU Medical Device Regulation (MDR, 2017/745). It is a read-only "
        f"tree of paths like {root}: root MDR, chapters, articles (Art-N), paragraphs, "
        "and annexes. Tools: search(query) finds passages, list(path) shows the children "
        "of a path, read(path) returns the full text below a path.\n"
        "Locate the relevant articles with search or list, then read the whole article "
        "before answering; do not answer from search snippets alone.\n"
        f"Answer in {LANGUAGE[lang]}, concisely. Cite the path of every statement in square "
        "brackets. Statements that are your own conclusion and not written in the text "
        f"end with {INFERRED[lang]}."
    )


def build_agent(conn: psycopg.Connection, llm: Llm, chat_model: BaseChatModel, lang: Lang):
    seen: dict[str, str] = {}  # path -> heading of everything the agent has read or found

    @tool("search")
    def search_tool(query: str) -> str:
        """Hybrid search; returns the best passages with their paths."""
        hits = search(conn, llm, query, lang, k=5)
        seen.update({h.path: h.heading for h in hits})
        return "\n".join(f"[{h.path}] {h.heading}: {h.text[:300]}" for h in hits)

    @tool("list")
    def list_tool(path: str = "MDR") -> str:
        """List the children of a path, e.g. MDR or MDR/Kapitel-II."""
        return "\n".join(tree.list_children(conn, lang, path)) or f"{path}: no children"

    @tool("read")
    def read_tool(path: str) -> str:
        """Read the full text of an article, paragraph, annex or annex section."""
        text, paths = tree.read(conn, lang, path)
        if paths:
            heading = text.split("\n", 1)[0]
            seen.update(dict.fromkeys([path.rstrip("/"), *paths], heading))
        return text

    tools = [search_tool, list_tool, read_tool]
    with_tools = chat_model.bind_tools(tools)

    def agent(state: State) -> dict:
        if state["steps"] >= MAX_STEPS:
            final = SystemMessage("Step limit reached. Answer now with what you have read.")
            return {"messages": [chat_model.invoke([*state["messages"], final])]}
        return {"messages": [with_tools.invoke(state["messages"])]}

    def route(state: State) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        has_read = any(isinstance(m, ToolMessage) and m.name == "read" for m in state["messages"])
        nudged = any(m.content == READ_FIRST for m in state["messages"])
        if has_read or nudged or state["steps"] >= MAX_STEPS:
            return END
        return "read_first"

    def read_first(state: State) -> dict:
        return {"messages": [HumanMessage(READ_FIRST)]}

    def count_step(state: State) -> dict:
        return {"steps": state["steps"] + 1}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("count", count_step)
    graph.add_node("read_first", read_first)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "read_first", END])
    graph.add_edge("read_first", "agent")
    graph.add_edge("tools", "count")
    graph.add_edge("count", "agent")
    return graph.compile(), seen


def run_agent(
    conn: psycopg.Connection,
    llm: Llm,
    chat_model: BaseChatModel,
    question: str,
    lang: Lang | None = None,
) -> AgentAnswer:
    lang = lang or detect_lang(question)
    graph, seen = build_agent(conn, llm, chat_model, lang)
    state = graph.invoke(
        {"messages": [SystemMessage(system_prompt(lang)), HumanMessage(question)], "steps": 0}
    )
    steps = [
        ToolCall(tool=call["name"], args=call["args"])
        for message in state["messages"]
        for call in getattr(message, "tool_calls", None) or []
    ]
    text, cited = verify_citations(state["messages"][-1].content, seen)
    return AgentAnswer(
        lang=lang,
        answer=text,
        citations=[citation(lang, path, seen[path]) for path in cited],
        steps=steps,
    )
