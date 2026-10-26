from mdr_rag.evaluate import first_relevant_rank, score


def test_metrics():
    ranks = [
        first_relevant_rank(["Art-1", "Art-10"], ["Art-10"]),  # 2
        first_relevant_rank(["Art-10"], ["Art-10", "Annex-II"]),  # 1
        first_relevant_rank(["Art-1"] * 10, ["Art-10"]),  # miss
        7,
    ]

    assert ranks[:3] == [2, 1, None]
    assert score(ranks) == {"recall@5": 0.5, "mrr@10": (1 / 2 + 1 + 1 / 7) / 4}
