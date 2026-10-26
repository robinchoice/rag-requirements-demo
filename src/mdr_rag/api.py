from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, model_validator

from mdr_rag import db
from mdr_rag.agent import AgentAnswer, run_agent
from mdr_rag.answer import Answer, Requirements, ask, extract_requirements, normalize_unit
from mdr_rag.config import load_settings
from mdr_rag.llm import Llm, OpenAiLlm
from mdr_rag.retrieval import Lang, Mode, detect_lang, search

ASKED = Counter("mdr_ask_total", "Answered questions by outcome", ["found"])


class SearchHit(BaseModel):
    path: str
    heading: str
    text: str
    score: float
    ranks: dict[str, int]
    similarity: float | None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["Welche Pflichten hat ein Hersteller?"])
    lang: Lang | None = Field(None, description="Detected from the question if omitted")


class ExtractRequest(BaseModel):
    article: str | None = Field(None, examples=["Art. 10"], description="Article or annex")
    topic: str | None = Field(None, examples=["Software als Medizinprodukt"])
    lang: Lang | None = Field(None, description="Detected from the topic if omitted, else de")

    @model_validator(mode="after")
    def one_of(self):
        if bool(self.article) == bool(self.topic):
            raise ValueError("give either article or topic")
        return self


def create_app(
    llm: Llm | None = None,
    chat_model: BaseChatModel | None = None,
    database_url: str | None = None,
) -> FastAPI:
    settings = load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = db.pool(database_url or settings.database_url)
        with app.state.pool.connection() as conn:
            db.migrate(conn, settings.embed_dim)
        app.state.llm = llm or OpenAiLlm(settings)
        app.state.chat_model = chat_model or ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.chat_model,
            temperature=0,
        )
        yield
        app.state.pool.close()

    app = FastAPI(
        title="MDR RAG demo",
        description="Hybrid search, grounded answers and requirement extraction over "
        "the EU Medical Device Regulation (2017/745). Not legal advice.",
        lifespan=lifespan,
    )

    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz(request: Request):
        try:
            with request.app.state.pool.connection(timeout=2) as conn:
                chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        except Exception as error:
            return JSONResponse({"status": "unavailable", "error": str(error)}, status_code=503)
        return {"status": "ok", "chunks": chunks}

    def connection(request: Request) -> Iterator[psycopg.Connection]:
        with request.app.state.pool.connection() as conn:
            yield conn

    def model(request: Request) -> Llm:
        return request.app.state.llm

    Conn = Annotated[psycopg.Connection, Depends(connection)]
    Model = Annotated[Llm, Depends(model)]

    @app.get("/search")
    def search_endpoint(
        conn: Conn,
        llm: Model,
        q: Annotated[str, Query(min_length=2, examples=["Qualitätsmanagementsystem"])],
        lang: Lang | None = None,
        mode: Mode = "hybrid",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[SearchHit]:
        hits = search(conn, llm, q, lang or detect_lang(q), k=k, mode=mode)
        return [SearchHit(**h.__dict__) for h in hits]

    @app.post("/ask")
    def ask_endpoint(conn: Conn, llm: Model, body: AskRequest) -> Answer:
        answer = ask(conn, llm, body.question, body.lang)
        ASKED.labels(found=str(answer.found).lower()).inc()
        return answer

    @app.post("/extract-requirements")
    def extract_endpoint(conn: Conn, llm: Model, body: ExtractRequest) -> Requirements:
        try:
            if body.article:
                unit = normalize_unit(body.article)
                if not unit:
                    raise HTTPException(422, "article must look like 'Art. 10' or 'Annex I'")
                return extract_requirements(conn, llm, body.lang or "de", unit=unit)
            lang = body.lang or detect_lang(body.topic)
            return extract_requirements(conn, llm, lang, topic=body.topic)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/agent")
    def agent_endpoint(conn: Conn, llm: Model, request: Request, body: AskRequest) -> AgentAnswer:
        """Research agent: searches, lists and reads whole articles before answering."""
        return run_agent(conn, llm, request.app.state.chat_model, body.question, body.lang)

    return app
