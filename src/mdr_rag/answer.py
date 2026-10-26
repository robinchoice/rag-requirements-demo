"""Grounded answers and requirement extraction on top of hybrid retrieval."""

import re
from collections.abc import Iterable
from typing import Literal

import psycopg
from pydantic import BaseModel, Field

from mdr_rag.llm import Llm
from mdr_rag.retrieval import Hit, Lang, detect_lang, search, unit_chunks

# Below this cosine similarity of the best vector hit the question is off-topic:
# answer "not found" without asking the model. Calibrated on bge-m3 (see README).
MIN_SIMILARITY = 0.45
CONTEXT_K = 6
BATCH_CHARS = 4000
MAX_BATCHES = 5

NOT_FOUND = "NOT_FOUND"
NOT_FOUND_TEXT = {
    "de": "Dazu habe ich in der MDR nichts gefunden.",
    "en": "I found nothing on this in the MDR.",
}
INFERRED = {"de": "(gefolgert)", "en": "(inferred)"}
LANGUAGE = {"de": "German", "en": "English"}


class Citation(BaseModel):
    path: str
    heading: str
    url: str


class Answer(BaseModel):
    lang: Lang
    found: bool
    answer: str
    citations: list[Citation]


def eurlex_url(lang: str, unit: str | None) -> str:
    url = f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:32017R0745"
    if not unit:
        return url
    kind, number = unit.split("-", 1)
    return f"{url}#{'art' if kind == 'Art' else 'anx'}_{number}"


def context_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.path}] {h.heading}\n{h.text}" for h in hits)


def ask(conn: psycopg.Connection, llm: Llm, question: str, lang: Lang | None = None) -> Answer:
    lang = lang or detect_lang(question)
    hits = search(conn, llm, question, lang, k=CONTEXT_K)
    best = max((h.similarity or 0.0 for h in hits), default=0.0)
    if best < MIN_SIMILARITY:
        return Answer(lang=lang, found=False, answer=NOT_FOUND_TEXT[lang], citations=[])

    system = (
        "You answer questions about the EU Medical Device Regulation (MDR, 2017/745) "
        "using only the excerpts below. Each excerpt starts with its path in square brackets.\n"
        f"- Answer in {LANGUAGE[lang]}, concisely.\n"
        "- After each statement cite the excerpt it comes from with its exact path in "
        "square brackets, e.g. [MDR/Kapitel-II/Art-10/Abs-9].\n"
        f"- If a statement is your own conclusion and not stated in the excerpts, "
        f"end it with {INFERRED[lang]}.\n"
        f"- If the excerpts do not answer the question, reply only with {NOT_FOUND}."
    )
    reply = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context_block(hits)}\n\nQuestion: {question}"},
        ]
    )
    if NOT_FOUND in reply:
        return Answer(lang=lang, found=False, answer=NOT_FOUND_TEXT[lang], citations=[])
    headings = {h.path: h.heading for h in hits}
    text, cited = verify_citations(reply, headings)
    citations = [citation(lang, path, headings[path]) for path in cited]
    return Answer(lang=lang, found=True, answer=text, citations=citations)


def verify_citations(text: str, allowed: Iterable[str]) -> tuple[str, list[str]]:
    """Keep citations of passages the model actually saw, drop invented ones."""
    allowed = set(allowed)
    cited: dict[str, None] = {}

    def check(match: re.Match) -> str:
        paths = [p.strip() for p in re.split(r"[,;]", match[1])]
        valid = [p for p in paths if p in allowed]
        cited.update(dict.fromkeys(valid))
        return f"[{', '.join(valid)}]" if valid else ""

    text = re.sub(r"\[([^\[\]]*MDR/[^\[\]]*)\]", check, text)
    text = re.sub(r" +([.,;])", r"\1", re.sub(r" {2,}", " ", text))
    return text.strip(), list(cited)


def unit_of(path: str) -> str | None:
    """MDR/Kapitel-II/Art-10/Abs-9 -> Art-10, MDR/Anhang-I/... -> Annex-I."""
    for segment in path.split("/"):
        if re.match(r"(Art|Anhang|Annex)-", segment):
            return segment.replace("Anhang-", "Annex-")
    return None


def citation(lang: str, path: str, heading: str) -> Citation:
    return Citation(path=path, heading=heading, url=eurlex_url(lang, unit_of(path)))


Addressee = Literal[
    "manufacturer",
    "authorised_representative",
    "importer",
    "distributor",
    "economic_operator",
    "notified_body",
    "competent_authority",
    "member_state",
    "commission",
    "other",
]


class ExtractedRequirement(BaseModel):
    statement: str = Field(description="One obligation, one sentence, close to the legal text")
    addressee: Addressee = Field(description="Who must fulfil the obligation")
    source: str = Field(description="Path of the excerpt the obligation comes from")


class ExtractedRequirements(BaseModel):
    requirements: list[ExtractedRequirement]


class Requirement(ExtractedRequirement):
    id: str
    url: str


class Requirements(BaseModel):
    lang: Lang
    requirements: list[Requirement]
    dropped: int = Field(description="Extracted items whose source was not in the context")
    truncated: bool = Field(description=f"Only the first {MAX_BATCHES} batches were processed")


def extract_requirements(
    conn: psycopg.Connection,
    llm: Llm,
    lang: Lang,
    unit: str | None = None,
    topic: str | None = None,
) -> Requirements:
    """Requirements of a whole article/annex (unit) or of the best passages for a topic."""
    hits = unit_chunks(conn, lang, unit) if unit else search(conn, llm, topic or "", lang, k=5)
    if not hits:
        raise LookupError(f"nothing found for {unit or topic!r}")
    batches = _batches(hits)
    system = (
        "Extract the binding requirements (obligations: shall/must, "
        "'müssen', present-tense duties) from the excerpts of the EU Medical Device "
        "Regulation below. One requirement per obligation; split enumerations into "
        "separate requirements. Keep the statement in the language of the excerpt "
        "and close to its wording. Skip definitions, recitals and empowerments of the "
        "Commission. 'source' is the exact path in square brackets before the excerpt."
    )
    requirements: list[Requirement] = []
    dropped = 0
    counter: dict[str, int] = {}
    for batch in batches[:MAX_BATCHES]:
        reply = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": context_block(batch)},
            ],
            schema=ExtractedRequirements,
        )
        units = {h.path: h.unit for h in batch}
        for item in ExtractedRequirements.model_validate_json(reply).requirements:
            if item.source not in units:
                dropped += 1
                continue
            counter[item.source] = counter.get(item.source, 0) + 1
            requirements.append(
                Requirement(
                    **item.model_dump(),
                    id=f"{item.source.removeprefix('MDR/')}#{counter[item.source]}",
                    url=eurlex_url(lang, units[item.source]),
                )
            )
    return Requirements(
        lang=lang,
        requirements=requirements,
        dropped=dropped,
        truncated=len(batches) > MAX_BATCHES,
    )


def _batches(hits: list[Hit]) -> list[list[Hit]]:
    batches: list[list[Hit]] = []
    size = 0
    for hit in hits:
        if not batches or size + len(hit.text) > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(hit)
        size += len(hit.text)
    return batches


def normalize_unit(text: str) -> str | None:
    """'Art. 10', 'Artikel 10', 'Annex I', 'Anhang VIII' -> 'Art-10', 'Annex-VIII'."""
    if m := re.fullmatch(r"\s*art(?:ikel|icle|\.)?\s*-?\s*(\d+)\s*", text, re.IGNORECASE):
        return f"Art-{m[1]}"
    if m := re.fullmatch(r"\s*(?:annex|anhang)\s*-?\s*([ivxlc]+)\s*", text, re.IGNORECASE):
        return f"Annex-{m[1].upper()}"
    return None
