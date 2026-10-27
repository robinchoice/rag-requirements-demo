"""Retrieval evaluation: recall@5 and MRR@10 for vector, full-text and hybrid search.

    uv run python -m mdr_rag.evaluate [eval/questions.toml]

Prints a Markdown table with a column per question kind (paraphrase, exact, all) and the
best vector similarity of on- and off-topic questions, which is what the "not found"
threshold in answer.py is calibrated on.
"""

import argparse
import datetime
import tomllib
from pathlib import Path

from mdr_rag import db
from mdr_rag.config import load_settings
from mdr_rag.llm import OpenAiLlm
from mdr_rag.retrieval import search, vector_search

MODES = ["vector", "fulltext", "hybrid"]
KINDS = ["paraphrase", "exact"]


def first_relevant_rank(units: list[str], expected: list[str]) -> int | None:
    return next((i for i, unit in enumerate(units, start=1) if unit in expected), None)


def score(ranks: list[int | None]) -> dict[str, float]:
    return {
        "recall@5": sum(1 for r in ranks if r and r <= 5) / len(ranks),
        "mrr@10": sum(1 / r for r in ranks if r and r <= 10) / len(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", type=Path, nargs="?", default=Path("eval/questions.toml"))
    args = parser.parse_args()
    questions = tomllib.loads(args.questions.read_text())["question"]
    on_topic = [q for q in questions if q["units"]]

    settings = load_settings()
    llm = OpenAiLlm(settings)
    ranks = {(kind, mode): [] for kind in KINDS for mode in MODES}
    best_similarity = {"on-topic": [], "off-topic": []}
    with db.connect(settings.database_url) as conn:
        for q in questions:
            embedding = llm.embed([q["q"]])[0]
            top = vector_search(conn, embedding, q["lang"], k=1)
            topic = "on-topic" if q["units"] else "off-topic"
            best_similarity[topic].append(top[0].similarity if top else 0.0)
            if not q["units"]:
                continue
            for mode in MODES:
                hits = search(conn, llm, q["q"], q["lang"], k=10, mode=mode)
                rank = first_relevant_rank([h.unit for h in hits], q["units"])
                ranks[(q.get("kind", "paraphrase"), mode)].append(rank)
                if mode == "hybrid" and not (rank and rank <= 5):
                    print(f"miss ({mode}): {q['q']} -> {[h.unit for h in hits[:5]]}")

    print(
        f"\n{len(on_topic)} questions ({sum(q['lang'] == 'de' for q in on_topic)} de, "
        f"{sum(q['lang'] == 'en' for q in on_topic)} en), embeddings: {settings.embed_model}, "
        f"{datetime.date.today().isoformat()}\n"
    )
    counts = [sum(q.get("kind", "paraphrase") == kind for q in on_topic) for kind in KINDS]
    columns = [f"{kind} ({n})" for kind, n in zip(KINDS, counts, strict=True)]
    columns.append(f"all ({len(on_topic)})")
    print("recall@5 / MRR@10\n")
    print(f"| Retriever | {' | '.join(columns)} |\n|---{'|---' * len(columns)}|")
    for mode in MODES:
        groups = [ranks[(kind, mode)] for kind in KINDS]
        groups.append([r for g in groups for r in g])
        cells = [score(g) for g in groups]
        print(
            f"| {mode} | "
            + " | ".join(f"{s['recall@5']:.2f} / {s['mrr@10']:.2f}" for s in cells)
            + " |"
        )
    for topic, values in best_similarity.items():
        print(f"\nbest vector similarity, {topic}: min {min(values):.2f}, max {max(values):.2f}")


if __name__ == "__main__":
    main()
