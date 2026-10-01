"""The evaluator contract.

Every evaluator -- deterministic metric or LLM judge -- implements the same
:class:`Evaluator` protocol and produces the same
:class:`~ai_eval_harness.results.EvaluationResult`. That uniformity is what lets
the runner, the aggregator, the gates, and the reports treat all evaluators
alike.

The uniformity stops at *execution*, not at *interpretation*. Each result carries
an :class:`~ai_eval_harness.results.EvaluatorKind`, and reports retain provenance
alongside separate evaluator aggregates, so a reader can always separate reproducible arithmetic
from a model's opinion. Running them through one pipeline is a convenience;
claiming they are the same kind of evidence would be a lie.

Subclasses implement :meth:`BaseEvaluator.measure`, which returns a
:class:`~ai_eval_harness.results.Measurement` and is free to raise. The base class
owns everything else -- threshold application, status assignment, error capture --
so those rules are defined once and cannot drift between evaluators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from ai_eval_harness.errors import ProviderError
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorKind,
    Measurement,
)

__all__ = ["BaseEvaluator", "Evaluator"]


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can score a case and report the outcome in a structured form."""

    @property
    def name(self) -> str:
        """Instance name, unique within a suite (e.g. ``recall@3``). Gates reference this."""
        ...

    @property
    def evaluator_type(self) -> str:
        """Registry type this instance was built from (e.g. ``recall_at_k``)."""
        ...

    @property
    def kind(self) -> EvaluatorKind:
        """Whether results are deterministic or model-generated."""
        ...

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate one case. Must not raise: failures become ``ERROR`` results."""
        ...


class BaseEvaluator(ABC):
    """Base class implementing the shared evaluator machinery.

    Args:
        min_case_score: Optional per-case threshold. When given, each scored
            case becomes ``PASSED`` (``score >= min_case_score``) or ``FAILED``,
            and the evaluator reports a pass rate. When ``None``, scored cases
            become ``SCORED`` and the pass rate is ``None`` rather than a
            misleading 100%.

    Raises:
        ValueError: If ``min_case_score`` is outside ``[0.0, 1.0]``.
    """

    _evaluator_type: ClassVar[str]
    """Registry key. Set by each concrete subclass.

    Exposed through the :attr:`evaluator_type` property rather than directly, so
    that a subclass serving several registry types (the LLM judge) can override
    it per instance without a class-attribute/property clash.
    """

    kind: ClassVar[EvaluatorKind]
    """Provenance of this evaluator's results. Set by each concrete subclass."""

    description: ClassVar[str]
    """One-line summary, shown by ``ai-eval list-evaluators``."""

    def __init__(self, *, min_case_score: float | None = None) -> None:
        if min_case_score is not None and not 0.0 <= min_case_score <= 1.0:
            raise ValueError(f"min_case_score must be within [0.0, 1.0], got {min_case_score}")
        self._min_case_score = min_case_score

    @property
    def evaluator_type(self) -> str:
        """The registry type this instance was built from."""
        return self._evaluator_type

    @property
    def name(self) -> str:
        """Instance name. Defaults to the registry type; subclasses with
        parameters override this so that, say, Recall@3 and Recall@5 are
        distinguishable in reports and addressable by separate gates."""
        return self.evaluator_type

    @property
    def min_case_score(self) -> float | None:
        """The configured per-case threshold, if any."""
        return self._min_case_score

    @abstractmethod
    def measure(self, case: EvaluationCase) -> Measurement:
        """Compute this evaluator's measurement for one case.

        Implementations return a ``Measurement`` whose ``score`` is ``None``
        when the metric is undefined for the case. They may raise; the base
        class converts any exception into an ``ERROR`` result.

        Args:
            case: The case to measure.

        Returns:
            The measurement.
        """

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate one case and build its result.

        Never raises. An exception from :meth:`measure` is captured as an
        ``ERROR`` result carrying the exception type and message, so one broken
        evaluator cannot abort a suite and cannot be mistaken for a zero score.

        Args:
            case: The case to evaluate.

        Returns:
            The structured result.
        """
        try:
            measurement = self.measure(case)
        except Exception as exc:
            return EvaluationResult(
                evaluator=self.name,
                evaluator_type=self.evaluator_type,
                kind=self.kind,
                case_id=case.case_id,
                status=EvaluationStatus.ERROR,
                score=None,
                threshold=self._min_case_score,
                explanation=f"Evaluator raised {type(exc).__name__}: {exc}",
                metadata={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    **({"retryable": exc.retryable} if isinstance(exc, ProviderError) else {}),
                },
            )

        status = self._status_for(measurement.score)
        return EvaluationResult(
            evaluator=self.name,
            evaluator_type=self.evaluator_type,
            kind=self.kind,
            case_id=case.case_id,
            status=status,
            score=measurement.score,
            threshold=self._min_case_score,
            explanation=measurement.explanation,
            metadata=dict(measurement.metadata),
        )

    def _status_for(self, score: float | None) -> EvaluationStatus:
        """Map a measurement's score onto a status. See :mod:`ai_eval_harness.results`."""
        if score is None:
            return EvaluationStatus.SKIPPED
        if self._min_case_score is None:
            return EvaluationStatus.SCORED
        return EvaluationStatus.PASSED if score >= self._min_case_score else EvaluationStatus.FAILED
