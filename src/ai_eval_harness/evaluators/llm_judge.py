"""The LLM-as-a-judge evaluator.

One class serves every judged criterion. It holds no prompt text and no provider
logic: it is given a prompt builder and a provider, and its only job is to run
the four-step pipeline -- build, call, parse, normalise -- and turn the result
into the same :class:`~ai_eval_harness.results.EvaluationResult` a deterministic
evaluator produces.

Failure handling is deliberate at every step:

* the builder declines (no answer, no context) -> ``SKIPPED``;
* the provider raises (unavailable, not recorded, transport failure) -> the
  exception propagates to :class:`~ai_eval_harness.evaluators.base.BaseEvaluator`,
  which records an ``ERROR``;
* the response will not parse -> ``ERROR``.

No branch produces a number that the judge did not actually give. That is the
whole point: a judge that silently defaults is indistinguishable from a judge
that is wrong.
"""

from __future__ import annotations

from typing import ClassVar

from ai_eval_harness.evaluators.base import BaseEvaluator
from ai_eval_harness.judges.base import JudgeProvider
from ai_eval_harness.judges.parsing import parse_verdict
from ai_eval_harness.judges.prompts import JudgePromptBuilder
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import EvaluatorKind, Measurement

__all__ = ["LlmJudgeEvaluator"]


class LlmJudgeEvaluator(BaseEvaluator):
    """Evaluates a case by asking a language model to judge it.

    Args:
        evaluator_name: Instance name used in reports and referenced by gates
            (for example ``llm_judge_groundedness``).
        criterion: The criterion being judged; recorded in metadata.
        prompt_builder: Builds the request, or declines with ``None``.
        provider: The judge provider to call.
        registry_type: The registry key this instance was built from.
        min_case_score: Optional per-case threshold.

    Note:
        Results from this evaluator carry ``kind = llm_judge``. They are not
        ground truth and are not reproducible in the way deterministic metrics
        are; see ``docs/LLM_JUDGES.md``.
    """

    kind: ClassVar[EvaluatorKind] = EvaluatorKind.LLM_JUDGE
    description: ClassVar[str] = (
        "Asks a configured judge provider to score a criterion. Requires a judge "
        "provider; introduces evaluator uncertainty."
    )

    def __init__(
        self,
        *,
        evaluator_name: str,
        criterion: str,
        prompt_builder: JudgePromptBuilder,
        provider: JudgeProvider,
        registry_type: str,
        min_case_score: float | None = None,
    ) -> None:
        super().__init__(min_case_score=min_case_score)
        self._name = evaluator_name
        self._criterion = criterion
        self._prompt_builder = prompt_builder
        self._provider = provider
        self._registry_type = registry_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def evaluator_type(self) -> str:
        # One class serves several registry types, so the type is per instance:
        # reports name the concrete type (llm_judge_groundedness) rather than the
        # shared implementation.
        return self._registry_type

    @property
    def criterion(self) -> str:
        """The criterion this instance judges."""
        return self._criterion

    @property
    def provider(self) -> JudgeProvider:
        """The provider this instance calls."""
        return self._provider

    def measure(self, case: EvaluationCase) -> Measurement:
        request = self._prompt_builder(case)
        if request is None:
            return Measurement(
                score=None,
                explanation=(
                    f"Case lacks the inputs required to judge {self._criterion!r} "
                    "(a generated answer, and retrieved document text where the "
                    "criterion needs it)."
                ),
                metadata={"criterion": self._criterion},
            )

        # Both of the following may raise; BaseEvaluator records that as ERROR.
        response = self._provider.complete(request)
        verdict = parse_verdict(response.text)

        return Measurement(
            score=verdict.score,
            explanation=f"Judge scored {verdict.score:.3f}: {verdict.reasoning}",
            metadata={
                "criterion": self._criterion,
                "judge_provider": self._provider.name,
                "judge_model": response.model,
                "prompt_version": request.metadata.get("prompt_version"),
                "judge_reasoning": verdict.reasoning,
            },
        )
