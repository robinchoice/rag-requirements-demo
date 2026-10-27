from conftest import FakeLlm

from mdr_rag.chunking import Chunk
from mdr_rag.ingest import load
from mdr_rag.retrieval import Hit, detect_lang, reciprocal_rank_fusion, references, search

CHUNKS = [
    Chunk(
        "de", "MDR/Kapitel-II/Art-10/Abs-9", "Art-10", "9", "Artikel 10 – Pflichten der Hersteller",
        "Die Hersteller unterhalten ein Qualitätsmanagementsystem.", 0,
    ),
    Chunk(
        "de", "MDR/Kapitel-II/Art-13/Abs-1", "Art-13", "1", "Artikel 13 – Pflichten der Importeure",
        "Importeure bringen nur Produkte in Verkehr, die dieser Verordnung entsprechen.", 1,
    ),
    Chunk(
        "de", "MDR/Anhang-I/Kapitel-II/Abschnitt-17", "Annex-I", "17", "Anhang I – Software",
        "Software wird nach dem Stand der Technik entwickelt und hergestellt.", 2,
    ),
    Chunk(
        "en", "MDR/Chapter-II/Art-10/Para-9", "Art-10", "9", "Article 10 – Obligations",
        "Manufacturers shall maintain a quality management system.", 0,
    ),
]  # fmt: skip


def hit(id: int) -> Hit:
    return Hit(id, "de", f"p{id}", "u", None, "h", "t")


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([[hit(1), hit(2), hit(3)], [hit(3), hit(4)]], k=60)

    assert [h.id for h in fused] == [3, 1, 2, 4]  # 3 is in both lists
    assert fused[0].score == 1 / 63 + 1 / 61


def test_search_modes_stay_within_the_language(conn):
    load(conn, FakeLlm(), "m", CHUNKS[:3])
    load(conn, FakeLlm(), "m", CHUNKS[3:])
    llm = FakeLlm()

    fulltext = search(conn, llm, "Welche Software-Anforderungen gibt es?", "de", mode="fulltext")
    vector = search(conn, llm, "Qualitätsmanagementsystem der Hersteller", "de", mode="vector")
    hybrid = search(conn, llm, "Pflichten der Importeure", "de", k=2)

    assert fulltext[0].path == "MDR/Anhang-I/Kapitel-II/Abschnitt-17"
    assert vector[0].path == "MDR/Kapitel-II/Art-10/Abs-9"
    assert hybrid[0].path == "MDR/Kapitel-II/Art-13/Abs-1"
    assert hybrid[0].ranks == {"vector": 1, "fulltext": 1}
    assert len(hybrid) == 2
    assert all(h.lang == "de" for h in fulltext + vector + hybrid)


def test_fulltext_without_matching_terms_returns_nothing(conn):
    load(conn, FakeLlm(), "m", CHUNKS)

    assert search(conn, FakeLlm(), "Brotbacken am Sonntag", "de", mode="fulltext") == []


def test_detect_lang():
    assert detect_lang("Welche Pflichten haben Hersteller?") == "de"
    assert detect_lang("What must manufacturers do?") == "en"


def test_references_named_in_a_question():
    assert references("Was regelt Artikel 120?") == (["Art-120"], [])
    assert references("Pflichten nach Art. 10 und Anhang IX") == (["Art-10", "Annex-IX"], [])
    assert references("Which products are listed in Annex XVI?") == (["Annex-XVI"], [])
    assert references("What does Rule 8 cover?") == ([], ["8"])
    assert references("Welche Pflichten haben Hersteller?") == ([], [])


RULES = [
    Chunk(
        "de", "MDR/Anhang-VIII/Kapitel-III/Abschnitt-6", "Annex-VIII", "6", "Anhang VIII – Regeln",
        "6. AKTIVE PRODUKTE 6.3. Regel 11 Software für Diagnosen gehört zur Klasse IIa.",
        3,
    ),
    Chunk(
        "de", "MDR/Anhang-VIII/Kapitel-III/Abschnitt-7", "Annex-VIII", "7", "Anhang VIII – Regeln",
        "7. BESONDERE REGELN 7.1. Regel 14 Produkte mit Arzneimitteln, anders als gemäß Regel 11.",
        4,
    ),
]  # fmt: skip


def test_named_references_come_first(conn):
    load(conn, FakeLlm(), "m", CHUNKS[:3] + RULES)
    llm = FakeLlm()

    rule = search(conn, llm, "Was besagt Regel 11?", "de", k=4)
    article = search(conn, llm, "Was regelt Artikel 13?", "de", k=4)
    unresolved = search(conn, llm, "Was regelt Artikel 13?", "de", k=4, resolve_references=False)

    # the rule's own section, not the one that only mentions it
    assert [h.path for h in rule if "reference" in h.ranks] == [RULES[0].path]
    assert rule[0].text.startswith("6.3. Regel 11 Software")
    assert article[0].path == "MDR/Kapitel-II/Art-13/Abs-1"
    assert article[0].ranks == {"reference": 1}
    assert not any("reference" in h.ranks for h in unresolved)
    assert len({h.id for h in article}) == len(article)
