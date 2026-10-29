"""Tests for the ranking metrics eval_retrieval.py reports.

Pure arithmetic over ranks: the harness itself needs the index and Cohere, so
only the part that turns ranks into numbers is checked here.
"""

from eval_retrieval import ranking_metrics


def test_a_miss_counts_against_every_metric():
    # Two found at rank 1, one at rank 2, one never retrieved.
    line = ranking_metrics([1, 1, 2, None], k=6)
    assert "Hit@1 50%" in line
    assert "Hit@3 75%" in line
    assert "Hit@6 75%" in line
    assert "MRR 0.625" in line  # (1 + 1 + 0.5 + 0) / 4


def test_cutoffs_do_not_repeat_when_k_is_three():
    line = ranking_metrics([1, 3], k=3)
    assert line.count("Hit@3") == 1


def test_no_answerable_cases():
    assert ranking_metrics([], k=6) == "no answerable cases"
