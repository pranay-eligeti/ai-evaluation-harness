"""Retrieval-quality evaluators: Recall@K, Precision@K, and reciprocal rank.

These adapt the pure functions in :mod:`ai_eval_harness.metrics.retrieval` to the
case model. The adaptation layer owns exactly one decision the metrics cannot
make for themselves: what to do when a case carries no retrieval annotation at
all (``expected_document_ids is None``). That is reported as ``SKIPPED`` -- the
case was never labelled, so its retrieval quality is unknown, not zero.

An explicitly empty annotation (``expected_document_ids == []``) is a different
statement -- "no document in this corpus answers this question" -- and is passed
through to the metrics, which each decide whether they remain defined. See
``docs/METRICS.md``.
"""

from __future__ import annotations

from typing import ClassVar

from ai_eval_harness.evaluators.base import BaseEvaluator
from ai_eval_harness.metrics.retrieval import (
    PrecisionDenominator,
    dedupe_preserving_order,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    top_k,
)
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import EvaluatorKind, Measurement

__all__ = [
    "PrecisionAtKEvaluator",
    "RecallAtKEvaluator",
    "ReciprocalRankEvaluator",
]

_UNANNOTATED = (
    "Case has no expected_document_ids annotation, so retrieval quality is "
    "unknown for this case (this is different from an annotation of [], which "
    "asserts that no document is relevant)."
)


def _retrieval_context(case: EvaluationCase) -> dict[str, object]:
    """Shared report metadata describing what was retrieved."""
    retrieved = case.retrieved_document_ids
    distinct = dedupe_preserving_order(retrieved)
    return {
        "retrieved_count": len(retrieved),
        "distinct_retrieved_count": len(distinct),
        "duplicates_collapsed": len(retrieved) - len(distinct),
    }


class RecallAtKEvaluator(BaseEvaluator):
    """Recall@K: the share of relevant documents found within the top K.

    Args:
        k: Cut-off rank, must be >= 1.
        min_case_score: Optional per-case threshold.

    Raises:
        ValueError: If ``k < 1`` or ``min_case_score`` is out of range.
    """

    _evaluator_type: ClassVar[str] = "recall_at_k"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Share of relevant documents retrieved within the top K. "
        "Skipped when no document is annotated relevant."
    )

    def __init__(self, *, k: int, min_case_score: float | None = None) -> None:
        super().__init__(min_case_score=min_case_score)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.expected_document_ids is None:
            return Measurement(score=None, explanation=_UNANNOTATED)

        relevant = set(case.expected_document_ids)
        metadata = _retrieval_context(case)
        metadata.update({"k": self.k, "relevant_ids": sorted(relevant)})

        score = recall_at_k(case.retrieved_document_ids, case.expected_document_ids, self.k)
        if score is None:
            return Measurement(
                score=None,
                explanation=(
                    "Recall@K is undefined: the case annotates no relevant documents, "
                    "so there is no denominator."
                ),
                metadata=metadata,
            )

        found = sorted(set(top_k(case.retrieved_document_ids, self.k)) & relevant)
        metadata["found_ids"] = found
        metadata["missed_ids"] = sorted(relevant - set(found))
        return Measurement(
            score=score,
            explanation=(
                f"Recall@{self.k} = {score:.3f}: found {len(found)} of "
                f"{len(relevant)} relevant document(s) in the top {self.k}."
            ),
            metadata=metadata,
        )


class PrecisionAtKEvaluator(BaseEvaluator):
    """Precision@K: the share of the top K results that are relevant.

    Args:
        k: Cut-off rank, must be >= 1.
        denominator: ``"retrieved"`` (default) divides by the number of distinct
            results actually returned; ``"k"`` divides by the fixed cut-off. See
            :func:`ai_eval_harness.metrics.retrieval.precision_at_k`.
        min_case_score: Optional per-case threshold.

    Raises:
        ValueError: If ``k < 1``, ``denominator`` is not a recognised convention,
            or ``min_case_score`` is out of range.
    """

    _evaluator_type: ClassVar[str] = "precision_at_k"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Share of the top-K results that are relevant. Denominator convention is "
        "configurable ('retrieved' or 'k')."
    )

    _VALID_DENOMINATORS: ClassVar[tuple[str, ...]] = ("retrieved", "k")

    def __init__(
        self,
        *,
        k: int,
        denominator: PrecisionDenominator = "retrieved",
        min_case_score: float | None = None,
    ) -> None:
        super().__init__(min_case_score=min_case_score)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if denominator not in self._VALID_DENOMINATORS:
            raise ValueError(
                f"denominator must be one of {self._VALID_DENOMINATORS}, got {denominator!r}"
            )
        self.k = k
        self.denominator: PrecisionDenominator = denominator

    @property
    def name(self) -> str:
        return f"precision@{self.k}"

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.expected_document_ids is None:
            return Measurement(score=None, explanation=_UNANNOTATED)

        relevant = set(case.expected_document_ids)
        effective = top_k(case.retrieved_document_ids, self.k)
        metadata = _retrieval_context(case)
        metadata.update(
            {
                "k": self.k,
                "denominator": self.denominator,
                "relevant_ids": sorted(relevant),
                "evaluated_ids": effective,
            }
        )

        score = precision_at_k(
            case.retrieved_document_ids,
            case.expected_document_ids,
            self.k,
            denominator=self.denominator,
        )
        if score is None:
            return Measurement(
                score=None,
                explanation=(
                    "Precision@K is undefined: nothing was retrieved and the "
                    "'retrieved' denominator convention is in use. Use "
                    "denominator='k' to score empty retrieval as 0.0."
                ),
                metadata=metadata,
            )

        hits = sorted(set(effective) & relevant)
        divisor = self.k if self.denominator == "k" else len(effective)
        metadata["hit_ids"] = hits
        return Measurement(
            score=score,
            explanation=(
                f"Precision@{self.k} = {score:.3f}: {len(hits)} of {divisor} "
                f"evaluated result(s) were relevant "
                f"(denominator convention: {self.denominator})."
            ),
            metadata=metadata,
        )


class ReciprocalRankEvaluator(BaseEvaluator):
    """Reciprocal rank of the first relevant document.

    The suite-level mean of this evaluator *is* Mean Reciprocal Rank. The
    per-case name is kept distinct so that a single case's value is never
    reported as "MRR".

    Args:
        k: Optional cut-off. Relevant documents below rank K score 0.0. ``None``
            searches the whole list.
        min_case_score: Optional per-case threshold.

    Raises:
        ValueError: If ``k`` is given and < 1, or ``min_case_score`` is out of range.
    """

    _evaluator_type: ClassVar[str] = "reciprocal_rank"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Reciprocal of the rank of the first relevant document; the suite mean of "
        "this evaluator is Mean Reciprocal Rank (MRR)."
    )

    def __init__(self, *, k: int | None = None, min_case_score: float | None = None) -> None:
        super().__init__(min_case_score=min_case_score)
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError(f"k must be >= 1 when provided, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return "reciprocal_rank" if self.k is None else f"reciprocal_rank@{self.k}"

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.expected_document_ids is None:
            return Measurement(score=None, explanation=_UNANNOTATED)

        relevant = set(case.expected_document_ids)
        metadata = _retrieval_context(case)
        metadata.update({"k": self.k, "relevant_ids": sorted(relevant)})

        score = reciprocal_rank(case.retrieved_document_ids, case.expected_document_ids, self.k)
        if score is None:
            return Measurement(
                score=None,
                explanation=(
                    "Reciprocal rank is undefined: the case annotates no relevant "
                    "documents, so there is no rank to find."
                ),
                metadata=metadata,
            )

        if score == 0.0:
            scope = "the result list" if self.k is None else f"the top {self.k}"
            return Measurement(
                score=0.0,
                explanation=f"No relevant document appears in {scope}; reciprocal rank = 0.000.",
                metadata=metadata,
            )

        rank = round(1.0 / score)
        metadata["first_relevant_rank"] = rank
        return Measurement(
            score=score,
            explanation=(
                f"First relevant document is at rank {rank}; reciprocal rank = {score:.3f}."
            ),
            metadata=metadata,
        )
