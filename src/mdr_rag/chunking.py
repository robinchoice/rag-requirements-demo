"""Turn the Docling document of the MDR into chunks along its legal structure.

EUR-Lex HTML becomes a flat stream of text items and one-row tables (the lettered
points) in Docling. Headings are recognised by their wording, so the same walk works
for both languages:

    articles: MDR/Kapitel-II/Art-10/Abs-9          MDR/Chapter-II/Art-10/Para-9
    annexes:  MDR/Anhang-I/Kapitel-II/Abschnitt-17 MDR/Annex-I/Chapter-II/Section-17

Recitals, footnotes and the correlation table (Annex XVII) are left out.
"""

import re
from dataclasses import dataclass, field

from docling_core.types.doc import DoclingDocument, TableItem, TextItem

MAX_CHARS = 3000

LANGS = {
    "de": {
        "chapter": r"KAPITEL ([IVXLC]+)",
        "subsection": r"ABSCHNITT (\d+)",
        "article": r"Artikel (\d+)",
        "annex": r"ANHANG ([IVXLC]+)",
        "annex_part": r"(KAPITEL|TEIL) ([IVXLC]+|[A-Z])",
        "paragraph": r"\((\d+)\)\s",
        "end_of_articles": r"Geschehen zu ",
        "labels": {
            "Kapitel": "Kapitel",
            "Teil": "Teil",
            "KAPITEL": "Kapitel",
            "TEIL": "Teil",
            "article": "Artikel",
            "annex": "Anhang",
            "para": "Abs",
            "section": "Abschnitt",
        },
    },
    "en": {
        "chapter": r"CHAPTER ([IVXLC]+)",
        "subsection": r"SECTION (\d+)",
        "article": r"Article (\d+)",
        "annex": r"ANNEX ([IVXLC]+)",
        "annex_part": r"(CHAPTER|PART) ([IVXLC]+|[A-Z])",
        "paragraph": r"(\d+)\.\s",
        "end_of_articles": r"Done at ",
        "labels": {
            "Kapitel": "Chapter",
            "Teil": "Part",
            "CHAPTER": "Chapter",
            "PART": "Part",
            "article": "Article",
            "annex": "Annex",
            "para": "Para",
            "section": "Section",
        },
    },
}
ANNEX_SECTION = r"(\d+)\.\s"
SKIPPED_ANNEXES = {"XVII"}


@dataclass
class Chunk:
    lang: str
    path: str  # localized, e.g. MDR/Kapitel-II/Art-10/Abs-9
    unit: str  # language-neutral article or annex, e.g. Art-10, Annex-I
    section: str | None  # paragraph of an article or top-level section of an annex
    heading: str  # e.g. "Artikel 10 – Allgemeine Pflichten der Hersteller"
    text: str
    position: int = 0

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.text}"


@dataclass
class _Open:
    path: str
    unit: str
    section: str | None
    heading: str
    blocks: list[str] = field(default_factory=list)


def blocks_of(doc: DoclingDocument) -> list[tuple[str, bool]]:
    """Flatten the document to (text, is_point); a table row becomes the point 'a) text'."""
    blocks = []
    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem):
            blocks.append((item.text.strip(), False))
        elif isinstance(item, TableItem):
            cells = [c.text.strip() for c in item.data.table_cells if c.text.strip()]
            blocks.append((" ".join(cells), True))
    return [b for b in blocks if b[0]]


def chunk_blocks(blocks: list[tuple[str, bool]], lang: str) -> list[Chunk]:
    rx = LANGS[lang]
    labels = rx["labels"]

    def full(pattern: str, text: str) -> re.Match | None:
        return re.fullmatch(pattern, text)

    chunks: list[Chunk] = []
    current: _Open | None = None
    in_body = False  # recitals come before the first chapter
    chapter = article = article_title = annex = annex_title = None
    annex_part = annex_part_title = None
    expect: str | None = None  # which title the next text block is

    def flush() -> None:
        nonlocal current
        if current and current.blocks:
            chunks.extend(_split(current, lang))
        current = None

    def open_article_chunk(section: str | None) -> _Open:
        path = f"MDR/{labels['Kapitel']}-{chapter}/Art-{article}"
        if section:
            path += f"/{labels['para']}-{section}"
        return _Open(
            path, f"Art-{article}", section, f"{labels['article']} {article} – {article_title}"
        )

    def open_annex_chunk(section: str | None) -> _Open:
        path = f"MDR/{labels['annex']}-{annex}"
        heading = f"{labels['annex']} {annex} – {annex_title}"
        if annex_part:
            path += f"/{annex_part}"
            heading += f" – {annex_part_title}"
        if section:
            path += f"/{labels['section']}-{section}"
        return _Open(path, f"Annex-{annex}", section, heading)

    for text, is_point in blocks:
        if is_point:
            if in_body and current:
                current.blocks.append(text)
            continue
        if expect:
            if expect == "article":
                article_title = text
                current = open_article_chunk(None)
            elif expect == "annex":
                annex_title = text
                current = open_annex_chunk(None)
            elif expect == "annex_part":
                if len(text) < 200 and not re.match(ANNEX_SECTION, text):
                    annex_part_title = text
                    current = open_annex_chunk(None)
                    expect = None
                    continue
                current = open_annex_chunk(None)
                expect = None  # no title: the block is content, handled below
            if expect:
                expect = None
                continue

        if m := full(rx["annex"], text):
            flush()
            in_body = m[1] not in SKIPPED_ANNEXES
            annex, annex_part, annex_part_title = m[1], None, None
            expect = "annex" if in_body else None
            continue
        if not in_body:
            if full(rx["chapter"], text) and annex is None:
                in_body = True
            else:
                continue
        if annex is None:
            if m := full(rx["chapter"], text):
                flush()
                chapter = m[1]
                expect = "title"
            elif full(rx["subsection"], text):
                flush()
                expect = "title"
            elif m := full(rx["article"], text):
                flush()
                article = m[1]
                expect = "article"
            elif re.match(rx["end_of_articles"], text):
                flush()
                in_body = False
            elif m := re.match(rx["paragraph"], text):
                flush()
                current = open_article_chunk(m[1])
                current.blocks.append(text)
            elif current:
                current.blocks.append(text)
        else:
            if m := full(rx["annex_part"], text):
                flush()
                annex_part = f"{labels[m[1]]}-{m[2]}"
                expect = "annex_part"
            elif m := re.match(ANNEX_SECTION, text):
                flush()
                current = open_annex_chunk(m[1])
                current.blocks.append(text)
            elif current:
                current.blocks.append(text)
    flush()

    for i, chunk in enumerate(chunks):
        chunk.position = i
    return chunks


def _split(chunk: _Open, lang: str) -> list[Chunk]:
    """Pack blocks into parts of at most MAX_CHARS; a single longer block stays whole."""
    parts: list[list[str]] = [[]]
    size = 0
    for block in chunk.blocks:
        if parts[-1] and size + len(block) > MAX_CHARS:
            parts.append([])
            size = 0
        parts[-1].append(block)
        size += len(block) + 1

    part_label = LANGS[lang]["labels"]["Teil"]
    result = []
    for n, blocks in enumerate(parts, start=1):
        path = chunk.path if len(parts) == 1 else f"{chunk.path}/{part_label}-{n}"
        result.append(
            Chunk(lang, path, chunk.unit, chunk.section, chunk.heading, "\n".join(blocks))
        )
    return result
