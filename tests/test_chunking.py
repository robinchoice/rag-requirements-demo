from pathlib import Path

from mdr_rag.chunking import MAX_CHARS, chunk_blocks
from mdr_rag.ingest import parse

FIXTURES = Path(__file__).parent / "fixtures"


def test_docling_html_is_chunked_along_the_legal_structure():
    chunks = parse(FIXTURES / "mdr_excerpt_de.html", "de")

    assert [c.path for c in chunks] == [
        "MDR/Kapitel-I/Art-2",
        "MDR/Kapitel-II/Art-10/Abs-1",
        "MDR/Kapitel-II/Art-10/Abs-9",
        "MDR/Anhang-I/Kapitel-II/Abschnitt-17",
    ]
    art_10_9 = chunks[2]
    assert (art_10_9.unit, art_10_9.section) == ("Art-10", "9")
    assert art_10_9.heading == "Artikel 10 – Allgemeine Pflichten der Hersteller"
    assert "a) ein Konzept zur Einhaltung" in art_10_9.text
    assert "b) das Risikomanagement" in art_10_9.text
    annex = chunks[3]
    assert annex.unit == "Annex-I"
    assert annex.heading.endswith("– ANFORDERUNGEN AN AUSLEGUNG UND HERSTELLUNG")
    assert "17.1. Software" in annex.text
    assert all("Erwägungsgrund" not in c.text and "Präsident" not in c.text for c in chunks)


def test_english_paragraphs_and_long_chunks_are_split():
    blocks = [
        ("CHAPTER II", False),
        ("MAKING AVAILABLE ON THE MARKET", False),
        ("Article 10", False),
        ("General obligations of manufacturers", False),
        ("1. Manufacturers shall ensure conformity.", False),
        ("(a) a point that belongs to paragraph 1", True),
        ("2. Manufacturers shall establish a risk management system.", False),
        ("x" * MAX_CHARS, False),
    ]

    chunks = chunk_blocks(blocks, "en")

    assert [c.path for c in chunks] == [
        "MDR/Chapter-II/Art-10/Para-1",
        "MDR/Chapter-II/Art-10/Para-2/Part-1",
        "MDR/Chapter-II/Art-10/Para-2/Part-2",
    ]
    assert chunks[0].text.endswith("(a) a point that belongs to paragraph 1")
    assert [c.position for c in chunks] == [0, 1, 2]
