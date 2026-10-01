from collections.abc import Callable
from typing import Any

import pytest

from ai_eval_harness.metrics import (
    citation_scores,
    citation_validity,
    exact_match,
    lexical_groundedness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        (["a", "b"], ["a", "b"], 1, 0.5),
        (["a", "a", "b"], ["a", "b", "b"], 2, 1),
        ([], ["a"], 4, 0),
        (["x"], ["a"], 10, 0),
        (["a"], ["a"], 10, 1),
        ([], [], 1, None),
    ],
)
def test_recall(retrieved: list[str], relevant: list[str], k: int, expected: float | None) -> None:
    assert recall_at_k(retrieved, relevant, k) == expected


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        (["x", "a"], ["a"], 2, 0.5),
        (["a", "a", "b"], ["a", "b"], 2, 1),
        ([], ["a"], 3, None),
        (["a"], [], 3, 0),
        (["a"], ["a"], 5, 1),
    ],
)
def test_precision(
    retrieved: list[str], relevant: list[str], k: int, expected: float | None
) -> None:
    assert precision_at_k(retrieved, relevant, k) == expected


def test_fixed_precision_and_mrr() -> None:
    assert precision_at_k(["a"], ["a"], 5, "k") == 0.2
    assert precision_at_k([], [], 5, "k") == 0
    assert reciprocal_rank(["x", "x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "a"], ["a"], 1) == 0
    assert reciprocal_rank([], ["a"]) == 0
    assert reciprocal_rank(["a"], []) is None
    assert mean_reciprocal_rank([1, 0.5, 0]) == 0.5
    assert mean_reciprocal_rank([]) is None


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, reciprocal_rank])
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_even_without_annotations(
    metric: Callable[..., float | None],
    k: object,
) -> None:
    with pytest.raises(ValueError):
        metric([], [], k)


def test_invalid_denominator() -> None:
    with pytest.raises(ValueError):
        precision_at_k([], [], 1, "invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expected", "actual", "f1"),
    [
        ([], [], 1),
        (["a"], [], 0),
        ([], ["a"], 0),
        (["a"], ["x"], 0),
        (["a", "a"], ["a", "a"], 1),
        (["a", "b"], ["a", "x"], 0.5),
    ],
)
def test_citations(expected: list[str], actual: list[str], f1: float) -> None:
    assert citation_scores(expected, actual).f1 == f1


def test_validity_is_not_correctness() -> None:
    assert citation_validity([], ["a"]) is None
    assert citation_validity(["a", "x", "a"], ["a"]) == 0.5
    assert citation_validity(["wrong"], ["wrong"]) == 1
    assert citation_scores(["right"], ["wrong"]).f1 == 0


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("The BLUE!", "blue", 1),
        ("", "", 1),
        ("", "blue", 0),
        ("red", "blue", 0),
        ("very large", "very very large", 0.8),
    ],
)
def test_answer_f1(answer: str, reference: str, expected: float) -> None:
    assert token_f1(answer, reference) == pytest.approx(expected)


def test_lexical_limitation() -> None:
    assert exact_match("The blue!", "blue") == 1
    assert exact_match("azure", "blue") == 0
    assert lexical_groundedness("beacon is not blue", ["beacon is blue"]) == 1
    assert lexical_groundedness("blue", []) is None
    assert lexical_groundedness("the is", ["blue"]) is None


@pytest.mark.parametrize("ids", ["abc", [" "], [2], [None]])
def test_malformed_ids(ids: Any) -> None:
    with pytest.raises(ValueError):
        recall_at_k(ids, ["a"], 1)
    with pytest.raises(ValueError):
        citation_scores(ids, [])


@pytest.mark.parametrize("value", [-1, 2, float("nan"), float("inf"), True])
def test_malformed_mrr_values(value: float) -> None:
    with pytest.raises(ValueError):
        mean_reciprocal_rank([value])
