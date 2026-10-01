"""Answer-quality evaluators based on surface comparison with a reference answer.

Both evaluators here need two things: a reference answer to compare against, and
a generated answer to compare. Either being ``None`` yields ``SKIPPED``, because
the comparison cannot be made.

A generated answer of ``""`` is treated differently: the system *did* respond, it
responded with nothing, and that is scored (0.0 against any non-empty reference).
Conflating "no answer recorded" with "answered emptily" would let a broken
capture pipeline masquerade as a quality regression, or vice versa.

These metrics reward lexical overlap and cannot recognise a correct paraphrase.
``docs/METRICS.md`` sets out when that is an acceptable signal and when it is not.
"""

from __future__ import annotations

from typing import ClassVar

from ai_eval_harness.evaluators.base import BaseEvaluator
from ai_eval_harness.metrics.text import exact_match, token_f1
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import EvaluatorKind, Measurement

__all__ = ["AnswerExactMatchEvaluator", "AnswerTokenF1Evaluator"]


def _skip_reason(case: EvaluationCase) -> str:
    """Explain which answer field is absent. Only called when at least one is ``None``."""
    if case.reference_answer is None and case.generated_answer is None:
        return "Case has neither a reference answer nor a generated answer."
    if case.reference_answer is None:
        return (
            "Case has no reference_answer, so the generated answer cannot be "
            "compared against ground truth."
        )
    return (
        "Case has no generated_answer, so there is nothing to compare "
        "(note: an empty string would be scored, not skipped)."
    )


class AnswerExactMatchEvaluator(BaseEvaluator):
    """Exact match between generated and reference answers, after normalisation.

    Normalisation (lowercasing, punctuation-to-space, article removal,
    whitespace collapsing) is documented in
    :mod:`ai_eval_harness.metrics.text`. Useful for short factoid answers; too
    brittle for anything longer, where :class:`AnswerTokenF1Evaluator` is the
    better deterministic signal.

    Args:
        min_case_score: Optional per-case threshold. For a binary metric, any
            threshold in ``(0.0, 1.0]`` means "must match exactly".
    """

    _evaluator_type: ClassVar[str] = "answer_exact_match"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "1.0 if the generated answer equals the reference after normalisation, "
        "else 0.0. Skipped when either answer is absent."
    )

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.reference_answer is None or case.generated_answer is None:
            return Measurement(score=None, explanation=_skip_reason(case))

        score = exact_match(case.generated_answer, case.reference_answer)
        verdict = "matches" if score == 1.0 else "does not match"
        return Measurement(
            score=score,
            explanation=f"Normalised generated answer {verdict} the reference answer.",
            metadata={
                "generated_answer": case.generated_answer,
                "reference_answer": case.reference_answer,
            },
        )


class AnswerTokenF1Evaluator(BaseEvaluator):
    """SQuAD-style token-overlap F1 between generated and reference answers.

    Args:
        min_case_score: Optional per-case threshold.
    """

    _evaluator_type: ClassVar[str] = "answer_token_f1"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Token-overlap F1 between generated and reference answers. Skipped when "
        "either answer is absent."
    )

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.reference_answer is None or case.generated_answer is None:
            return Measurement(score=None, explanation=_skip_reason(case))

        score = token_f1(case.generated_answer, case.reference_answer)
        return Measurement(
            score=score,
            explanation=(
                f"Token-overlap F1 = {score:.3f} between the generated answer and "
                "the reference answer."
            ),
            metadata={
                "generated_answer": case.generated_answer,
                "reference_answer": case.reference_answer,
            },
        )
