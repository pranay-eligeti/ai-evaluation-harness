"""The result contract shared by every evaluator.

Deterministic metrics and LLM judges produce the *same* result type so the runner,
aggregator, gates, and reports can treat them uniformly -- but the result carries
an :class:`EvaluatorKind` so that consumers are never misled into believing a
model-generated judgement is ground truth. Uniform plumbing, explicit provenance.

Status semantics
----------------

``PASSED`` / ``FAILED``
    A score was produced *and* the evaluator was configured with a per-case
    threshold (``min_case_score``), so the score could be judged. Only these two
    statuses contribute to a pass rate.
``SCORED``
    A score was produced but no per-case threshold was configured. The number is
    real and is included in the mean; there is simply no per-case verdict. Pass
    rate is ``None`` for such evaluators rather than a misleading 100%.
``SKIPPED``
    The metric is *undefined* for this case -- typically because the required
    annotation is absent (``reference_answer=None``) or the metric has no
    denominator (Recall@K with no relevant documents). Skipped cases contribute
    no score. They are counted, reported, and can be gated on via
    ``max_skipped_count``; they are never silently treated as 0.0.
``ERROR``
    The evaluator raised. The failure is recorded with its type and message.
    Errors contribute no score and can be gated on via ``max_error_count``.

The ``SKIPPED``/``SCORED`` split is the mechanism that keeps the harness honest:
missing ground truth degrades *coverage*, which is visible, instead of degrading
*scores*, which would look like a quality regression that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_eval_harness.metrics.validation import validate_score

__all__ = [
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluatorKind",
    "Measurement",
]


class EvaluatorKind(StrEnum):
    """How a result was produced.

    This is recorded on every result and in every report. It exists so that a
    reader can always tell reproducible arithmetic apart from a model's opinion.
    """

    DETERMINISTIC = "deterministic"
    """Computed by a pure function. Identical inputs always give identical output."""

    LLM_JUDGE = "llm_judge"
    """Produced by a language model acting as a judge. Carries evaluator
    uncertainty: it may vary between runs, models, and prompt revisions.
    See ``docs/LLM_JUDGES.md``."""


class EvaluationStatus(StrEnum):
    """Outcome of evaluating one case with one evaluator. See the module docstring."""

    PASSED = "passed"
    FAILED = "failed"
    SCORED = "scored"
    SKIPPED = "skipped"
    ERROR = "error"

    @property
    def has_verdict(self) -> bool:
        """True when the status represents a per-case pass/fail judgement."""
        return self in (EvaluationStatus.PASSED, EvaluationStatus.FAILED)

    @property
    def is_scored(self) -> bool:
        """True when the status carries a usable numeric score."""
        return self in (
            EvaluationStatus.PASSED,
            EvaluationStatus.FAILED,
            EvaluationStatus.SCORED,
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """What an evaluator computes for one case, before status is assigned.

    Evaluators return a ``Measurement``; the base evaluator turns it into an
    :class:`EvaluationResult` by applying the configured threshold. Keeping the
    two apart means individual evaluators never have to reimplement (or
    accidentally vary) the status rules.

    Attributes:
        score: The metric value, conventionally in ``[0.0, 1.0]``. ``None``
            means the metric is *undefined* for this case, which becomes
            ``SKIPPED``. It never means "zero".
        explanation: Why this value was produced, in human terms. Required --
            a score with no explanation is not debuggable.
        metadata: Structured detail backing the explanation (matched ids,
            counts, sub-scores). Included verbatim in the report.
    """

    score: float | None
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None:
            validate_score(self.score)
        if not self.explanation.strip():
            raise ValueError("Measurement.explanation must not be blank")


class EvaluationResult(BaseModel):
    """The structured outcome of applying one evaluator to one case.

    Attributes:
        evaluator: Instance name of the evaluator, unique within a suite
            (for example ``recall@3``). Quality gates reference this name.
        evaluator_type: Registry type the instance was built from
            (for example ``recall_at_k``).
        kind: Deterministic or LLM-judge. See :class:`EvaluatorKind`.
        case_id: The case this result belongs to.
        status: See :class:`EvaluationStatus`.
        score: The numeric score, or ``None`` for ``SKIPPED``/``ERROR``.
        threshold: The per-case threshold applied, or ``None`` if the evaluator
            was configured without one (status is then ``SCORED``).
        explanation: Human-readable justification for the outcome.
        metadata: Structured supporting detail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator: str = Field(min_length=1)
    evaluator_type: str = Field(min_length=1)
    kind: EvaluatorKind
    case_id: str = Field(min_length=1)
    status: EvaluationStatus
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)
    threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)
    explanation: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> EvaluationResult:
        if self.status.is_scored != (self.score is not None):
            raise ValueError("Scored statuses require scores; skipped/error statuses forbid scores")
        if self.status.has_verdict:
            if self.threshold is None or self.score is None:
                raise ValueError("Verdict requires score and threshold")
            if (self.score >= self.threshold) != (self.status == EvaluationStatus.PASSED):
                raise ValueError("Verdict disagrees with score and threshold")
        if self.status == EvaluationStatus.SCORED and self.threshold is not None:
            raise ValueError("Scored status cannot have a threshold")
        if not self.explanation.strip():
            raise ValueError("Explanation must not be blank")
        return self
