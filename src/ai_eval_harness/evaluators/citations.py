"""Citation evaluators: correctness against ground truth, and validity against context.

See :mod:`ai_eval_harness.metrics.citations` for why these are two separate
questions rather than one combined "citation score".
"""

from __future__ import annotations

from typing import ClassVar

from ai_eval_harness.evaluators.base import BaseEvaluator
from ai_eval_harness.metrics.citations import citation_scores, citation_validity
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import EvaluatorKind, Measurement

__all__ = ["CitationF1Evaluator", "CitationValidityEvaluator"]


class CitationF1Evaluator(BaseEvaluator):
    """Citation correctness: F1 of cited ids against the ids that should be cited.

    The headline score is F1; precision and recall are carried in metadata so a
    failure can be diagnosed as over-citing or under-citing without a second run.

    Args:
        min_case_score: Optional per-case threshold.
    """

    _evaluator_type: ClassVar[str] = "citation_f1"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "F1 of cited source ids against expected_citations. Skipped when the case "
        "has no citation annotation."
    )

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.expected_citations is None:
            return Measurement(
                score=None,
                explanation=(
                    "Case has no expected_citations annotation, so citation "
                    "correctness is unknown (distinct from an annotation of [], "
                    "which asserts the answer should cite nothing)."
                ),
            )

        scores = citation_scores(case.expected_citations, case.actual_citations)
        metadata = {
            "precision": scores.precision,
            "recall": scores.recall,
            "expected_citations": sorted(set(case.expected_citations)),
            "actual_citations": sorted(set(case.actual_citations)),
            "matched": list(scores.matched),
            "spurious": list(scores.spurious),
            "missing": list(scores.missing),
        }

        if not scores.spurious and not scores.missing:
            detail = "all expected citations present and no spurious ones"
        else:
            parts = []
            if scores.missing:
                parts.append(f"missing {list(scores.missing)}")
            if scores.spurious:
                parts.append(f"spurious {list(scores.spurious)}")
            detail = "; ".join(parts)

        return Measurement(
            score=scores.f1,
            explanation=(
                f"Citation F1 = {scores.f1:.3f} "
                f"(precision {scores.precision:.3f}, recall {scores.recall:.3f}): {detail}."
            ),
            metadata=metadata,
        )


class CitationValidityEvaluator(BaseEvaluator):
    """Citation validity: do the cited ids resolve to documents that were retrieved?

    Requires no ground truth, so it runs on every case that cites something --
    including cases with no annotation at all. This is the check that catches a
    fabricated source identifier.

    Args:
        min_case_score: Optional per-case threshold.
    """

    _evaluator_type: ClassVar[str] = "citation_validity"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Share of cited ids that resolve to a retrieved document. Needs no ground "
        "truth. Skipped when the answer cites nothing."
    )

    def measure(self, case: EvaluationCase) -> Measurement:
        retrieved_ids = case.retrieved_document_ids
        score = citation_validity(case.actual_citations, retrieved_ids)
        if score is None:
            return Measurement(
                score=None,
                explanation=(
                    "Answer cites no sources, so citation validity is undefined. "
                    "Whether it should have cited sources is measured by citation_f1."
                ),
                metadata={"retrieved_ids": sorted(set(retrieved_ids))},
            )

        cited = set(case.actual_citations)
        unresolved = sorted(cited - set(retrieved_ids))
        metadata = {
            "actual_citations": sorted(cited),
            "retrieved_ids": sorted(set(retrieved_ids)),
            "unresolved_citations": unresolved,
        }
        if unresolved:
            detail = (
                f"{len(unresolved)} of {len(cited)} cited id(s) do not appear in the "
                f"retrieved context: {unresolved}"
            )
        else:
            detail = f"all {len(cited)} cited id(s) resolve to retrieved documents"
        return Measurement(
            score=score,
            explanation=f"Citation validity = {score:.3f}: {detail}.",
            metadata=metadata,
        )
