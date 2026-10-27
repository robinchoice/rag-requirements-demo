"""Grounded answers and requirement extraction on top of hybrid retrieval."""

import re
from collections.abc import Iterable
from typing import Literal

import psycopg
from pydantic import BaseModel, Field

from mdr_rag.llm import Llm
from mdr_rag.retrieval import Hit, Lang, detect_lang, search, unit_chunks

# Below this cosine similarity of the best vector hit the question is off-topic:
# answer "not found" without asking the model. bge-m3 on the eval set: off-topic
# questions reach at most 0.43, on-topic ones at least 0.60.
MIN_SIMILARITY = 0.5
CONTEXT_K = 4
EXCERPT_CHARS = 2000
BATCH_CHARS = 2500
MAX_BATCHES = 5
MAX_PER_BATCH = 12

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


def numbered_excerpts(hits: list[Hit], limit: int | None = None) -> str:
    """Excerpts as [1], [2], ...: small models cite short numbers far more reliably
    than long paths; the numbers are mapped back to paths afterwards."""
    return "\n\n".join(f"[{i}] {h.heading}\n{h.text[:limit]}" for i, h in enumerate(hits, start=1))


def ask(conn: psycopg.Connection, llm: Llm, question: str, lang: Lang | None = None) -> Answer:
    lang = lang or detect_lang(question)
    hits = search(conn, llm, question, lang, k=CONTEXT_K)
    best = max((h.similarity or 0.0 for h in hits), default=0.0)
    if best < MIN_SIMILARITY:
        return Answer(lang=lang, found=False, answer=NOT_FOUND_TEXT[lang], citations=[])

    system = (
        "You answer questions about the EU Medical Device Regulation (MDR, 2017/745) "
        "using only the numbered excerpts below.\n"
        f"- Answer in {LANGUAGE[lang]} in one to four sentences. Do not copy the excerpts.\n"
        "- After each sentence put the number of the excerpt that supports it in square "
        "brackets, e.g. [2].\n"
        f"- If a sentence is your own conclusion and not stated in the excerpts, "
        f"end it with {INFERRED[lang]}.\n"
        f"- If the excerpts do not answer the question, reply only with {NOT_FOUND}."
    )
    excerpts = numbered_excerpts(hits, EXCERPT_CHARS)
    reply = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{excerpts}\n\nQuestion: {question}"},
        ]
    )
    if NOT_FOUND in reply:
        return Answer(lang=lang, found=False, answer=NOT_FOUND_TEXT[lang], citations=[])
    text, cited = resolve_citations(reply, hits)
    citations = [citation(lang, h.path, h.heading) for h in cited]
    return Answer(lang=lang, found=True, answer=text, citations=citations)


def resolve_citations(text: str, hits: list[Hit]) -> tuple[str, list[Hit]]:
    """Turn [2] into [path of excerpt 2]; numbers without an excerpt are dropped."""
    cited: dict[int, Hit] = {}

    def resolve(match: re.Match) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match[1])]
        valid = [n for n in numbers if 1 <= n <= len(hits)]
        cited.update({n: hits[n - 1] for n in valid})
        return f"[{', '.join(hits[n - 1].path for n in valid)}]" if valid else ""

    text = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", resolve, text)
    return _tidy(text), list(cited.values())


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
    return _tidy(text), list(cited)


def _tidy(text: str) -> str:
    """Remove the gaps that dropped citations leave behind."""
    return re.sub(r" +([.,;])", r"\1", re.sub(r" {2,}", " ", text)).strip()


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


def extraction_schema(excerpts: int) -> type[BaseModel]:
    """Response schema; 'excerpt' is an enum of the excerpt numbers actually sent, so
    constrained decoding cannot produce a reference to nothing."""

    class ExtractedRequirement(BaseModel):
        statement: str = Field(
            max_length=600, description="One obligation, one sentence, close to the text"
        )
        addressee: Addressee = Field(description="Who must fulfil the obligation")
        excerpt: Literal[tuple(range(1, excerpts + 1))] = Field(  # type: ignore[valid-type]
            description="Number of the excerpt the obligation comes from"
        )

    class ExtractedRequirements(BaseModel):
        requirements: list[ExtractedRequirement] = Field(max_length=MAX_PER_BATCH)

    return ExtractedRequirements


class Requirement(BaseModel):
    id: str = Field(description="Source path plus running number, stable across runs")
    statement: str
    addressee: Addressee
    source: str
    url: str


class Requirements(BaseModel):
    lang: Lang
    requirements: list[Requirement]
    truncated: bool = Field(description=f"Only the first {MAX_BATCHES} batches were processed")


def extract_requirements(
    conn: psycopg.Connection,
    llm: Llm,
    lang: Lang,
    unit: str | None = None,
    topic: str | None = None,
) -> Requirements:
    """Requirements of a whole article/annex (unit) or of the best passages for a topic."""
    hits = (
        unit_chunks(conn, lang, unit) if unit else search(conn, llm, topic or "", lang, k=CONTEXT_K)
    )
    if not hits:
        raise LookupError(f"nothing found for {unit or topic!r}")
    batches = _batches(hits)
    system = (
        "Extract the binding requirements (obligations: shall/must, "
        "'müssen', present-tense duties) from the excerpts of the EU Medical Device "
        "Regulation below. One requirement per obligation; split enumerations into "
        "separate requirements. Keep the statement in the language of the excerpt "
        "and close to its wording. Skip definitions, recitals and empowerments of the "
        "Commission. 'addressee' is who has to act; requirements on the design and "
        "manufacture of devices address the manufacturer. 'excerpt' is the number in "
        "square brackets before the excerpt."
    )
    requirements: list[Requirement] = []
    counter: dict[str, int] = {}
    for batch in batches[:MAX_BATCHES]:
        schema = extraction_schema(len(batch))
        reply = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": numbered_excerpts(batch)},
            ],
            schema=schema,
            max_tokens=2048,
        )
        for item in schema.model_validate_json(reply).requirements:
            hit = batch[item.excerpt - 1]
            counter[hit.path] = counter.get(hit.path, 0) + 1
            requirements.append(
                Requirement(
                    id=f"{hit.path.removeprefix('MDR/')}#{counter[hit.path]}",
                    statement=item.statement,
                    addressee=item.addressee,
                    source=hit.path,
                    url=eurlex_url(lang, hit.unit),
                )
            )
    return Requirements(
        lang=lang,
        requirements=requirements,
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
