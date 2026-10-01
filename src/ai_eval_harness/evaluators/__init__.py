"""Evaluators: the layer that turns metrics and judges into structured results.

An evaluator answers one question about one case and reports the answer in a
uniform shape. Deterministic evaluators wrap the pure functions in
:mod:`ai_eval_harness.metrics`; the LLM-judge evaluator wraps the provider
abstraction in :mod:`ai_eval_harness.judges`. Both implement
:class:`~ai_eval_harness.evaluators.base.Evaluator` and both produce
:class:`~ai_eval_harness.results.EvaluationResult`.

Extending the harness with a new evaluator means: subclass
:class:`~ai_eval_harness.evaluators.base.BaseEvaluator`, implement ``measure``,
and add one entry to :mod:`ai_eval_harness.evaluators.registry`. Nothing in the
runner, the gates, the reports, or the CLI needs to change.
"""

from ai_eval_harness.evaluators.answer import (
    AnswerExactMatchEvaluator,
    AnswerTokenF1Evaluator,
)
from ai_eval_harness.evaluators.base import BaseEvaluator, Evaluator
from ai_eval_harness.evaluators.citations import (
    CitationF1Evaluator,
    CitationValidityEvaluator,
)
from ai_eval_harness.evaluators.groundedness import LexicalGroundednessEvaluator
from ai_eval_harness.evaluators.llm_judge import LlmJudgeEvaluator
from ai_eval_harness.evaluators.registry import (
    EvaluatorBuildContext,
    EvaluatorParameter,
    EvaluatorSpecification,
    available_evaluator_types,
    build_evaluator,
    get_specification,
    iter_specifications,
)
from ai_eval_harness.evaluators.retrieval import (
    PrecisionAtKEvaluator,
    RecallAtKEvaluator,
    ReciprocalRankEvaluator,
)

__all__ = [
    "AnswerExactMatchEvaluator",
    "AnswerTokenF1Evaluator",
    "BaseEvaluator",
    "CitationF1Evaluator",
    "CitationValidityEvaluator",
    "Evaluator",
    "EvaluatorBuildContext",
    "EvaluatorParameter",
    "EvaluatorSpecification",
    "LexicalGroundednessEvaluator",
    "LlmJudgeEvaluator",
    "PrecisionAtKEvaluator",
    "RecallAtKEvaluator",
    "ReciprocalRankEvaluator",
    "available_evaluator_types",
    "build_evaluator",
    "get_specification",
    "iter_specifications",
]
