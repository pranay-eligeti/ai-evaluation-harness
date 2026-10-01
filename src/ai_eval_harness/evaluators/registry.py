"""The evaluator registry: configuration strings to evaluator instances.

The registry is the single place that knows which evaluator types exist, what
parameters each accepts, and which ones need a judge provider. Keeping that in
one table means ``ai-eval list-evaluators`` and the configuration validator can
never disagree with what ``ai-eval run`` actually supports -- they read the same
data.

Configuration is validated strictly. An unrecognised type, an unrecognised
parameter, a missing required parameter, or a parameter of the wrong type is a
:class:`~ai_eval_harness.errors.ConfigError`, not a warning and not a default.
A silently-ignored ``k`` would mean silently evaluating at the wrong cut-off and
reporting the result as if it were the requested one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ai_eval_harness.errors import ConfigError, UnknownEvaluatorError
from ai_eval_harness.evaluators.answer import (
    AnswerExactMatchEvaluator,
    AnswerTokenF1Evaluator,
)
from ai_eval_harness.evaluators.base import Evaluator
from ai_eval_harness.evaluators.citations import (
    CitationF1Evaluator,
    CitationValidityEvaluator,
)
from ai_eval_harness.evaluators.groundedness import LexicalGroundednessEvaluator
from ai_eval_harness.evaluators.llm_judge import LlmJudgeEvaluator
from ai_eval_harness.evaluators.retrieval import (
    PrecisionAtKEvaluator,
    RecallAtKEvaluator,
    ReciprocalRankEvaluator,
)
from ai_eval_harness.judges.base import JudgeProvider
from ai_eval_harness.judges.prompts import (
    ANSWER_RELEVANCE_CRITERION,
    GROUNDEDNESS_CRITERION,
    build_answer_relevance_request,
    build_groundedness_request,
)
from ai_eval_harness.results import EvaluatorKind

__all__ = [
    "EvaluatorBuildContext",
    "EvaluatorParameter",
    "EvaluatorSpecification",
    "available_evaluator_types",
    "build_evaluator",
    "get_specification",
    "iter_specifications",
]


@dataclass(frozen=True, slots=True)
class EvaluatorBuildContext:
    """Resources an evaluator may need that are not part of its own parameters.

    Attributes:
        judge_provider: The configured provider, or ``None`` if no judge is
            configured. Evaluator types with ``requires_judge`` fail to build
            when this is ``None``, rather than silently degrading to a
            deterministic approximation.
    """

    judge_provider: JudgeProvider | None = None


@dataclass(frozen=True, slots=True)
class EvaluatorParameter:
    """One configurable parameter of an evaluator type."""

    name: str
    description: str
    required: bool = False
    default: Any = None


EvaluatorFactory = Callable[[Mapping[str, Any], "float | None", EvaluatorBuildContext], Evaluator]


@dataclass(frozen=True, slots=True)
class EvaluatorSpecification:
    """Everything the harness knows about one evaluator type."""

    type: str
    kind: EvaluatorKind
    description: str
    factory: EvaluatorFactory
    parameters: tuple[EvaluatorParameter, ...] = field(default_factory=tuple)
    requires_judge: bool = False


# --------------------------------------------------------------------------- #
# Parameter coercion
# --------------------------------------------------------------------------- #


def _int_param(params: Mapping[str, Any], name: str, evaluator_type: str) -> int:
    value = params[name]
    # bool is a subclass of int; `k = true` is a configuration mistake, not k=1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"Evaluator {evaluator_type!r}: parameter {name!r} must be an integer, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _optional_int_param(params: Mapping[str, Any], name: str, evaluator_type: str) -> int | None:
    if name not in params or params[name] is None:
        return None
    return _int_param(params, name, evaluator_type)


def _str_param(params: Mapping[str, Any], name: str, evaluator_type: str, default: str) -> str:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, str):
        raise ConfigError(
            f"Evaluator {evaluator_type!r}: parameter {name!r} must be a string, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _require_judge(context: EvaluatorBuildContext, evaluator_type: str) -> JudgeProvider:
    if context.judge_provider is None:
        raise ConfigError(
            f"Evaluator {evaluator_type!r} requires a judge provider, but none is "
            "configured. Add a [judge] section to the suite configuration, or "
            "remove this evaluator. The harness will not substitute a "
            "deterministic heuristic for a semantic judgement."
        )
    return context.judge_provider


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def _build_recall_at_k(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del context
    return RecallAtKEvaluator(
        k=_int_param(params, "k", "recall_at_k"), min_case_score=min_case_score
    )


def _build_precision_at_k(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del context
    denominator = _str_param(params, "denominator", "precision_at_k", "retrieved")
    if denominator not in ("retrieved", "k"):
        raise ConfigError(
            "Evaluator 'precision_at_k': parameter 'denominator' must be "
            f"'retrieved' or 'k', got {denominator!r}"
        )
    return PrecisionAtKEvaluator(
        k=_int_param(params, "k", "precision_at_k"),
        denominator=cast(Literal["retrieved", "k"], denominator),
        min_case_score=min_case_score,
    )


def _build_reciprocal_rank(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del context
    return ReciprocalRankEvaluator(
        k=_optional_int_param(params, "k", "reciprocal_rank"),
        min_case_score=min_case_score,
    )


def _build_citation_f1(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params, context
    return CitationF1Evaluator(min_case_score=min_case_score)


def _build_citation_validity(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params, context
    return CitationValidityEvaluator(min_case_score=min_case_score)


def _build_answer_exact_match(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params, context
    return AnswerExactMatchEvaluator(min_case_score=min_case_score)


def _build_answer_token_f1(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params, context
    return AnswerTokenF1Evaluator(min_case_score=min_case_score)


def _build_lexical_groundedness(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params, context
    return LexicalGroundednessEvaluator(min_case_score=min_case_score)


def _build_judge_groundedness(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params
    return LlmJudgeEvaluator(
        evaluator_name="llm_judge_groundedness",
        criterion=GROUNDEDNESS_CRITERION,
        prompt_builder=build_groundedness_request,
        provider=_require_judge(context, "llm_judge_groundedness"),
        registry_type="llm_judge_groundedness",
        min_case_score=min_case_score,
    )


def _build_judge_answer_relevance(
    params: Mapping[str, Any], min_case_score: float | None, context: EvaluatorBuildContext
) -> Evaluator:
    del params
    return LlmJudgeEvaluator(
        evaluator_name="llm_judge_answer_relevance",
        criterion=ANSWER_RELEVANCE_CRITERION,
        prompt_builder=build_answer_relevance_request,
        provider=_require_judge(context, "llm_judge_answer_relevance"),
        registry_type="llm_judge_answer_relevance",
        min_case_score=min_case_score,
    )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

_K_PARAMETER = EvaluatorParameter(
    name="k", description="Cut-off rank; must be >= 1.", required=True
)

_REGISTRY: dict[str, EvaluatorSpecification] = {
    specification.type: specification
    for specification in (
        EvaluatorSpecification(
            type="recall_at_k",
            kind=RecallAtKEvaluator.kind,
            description=RecallAtKEvaluator.description,
            factory=_build_recall_at_k,
            parameters=(_K_PARAMETER,),
        ),
        EvaluatorSpecification(
            type="precision_at_k",
            kind=PrecisionAtKEvaluator.kind,
            description=PrecisionAtKEvaluator.description,
            factory=_build_precision_at_k,
            parameters=(
                _K_PARAMETER,
                EvaluatorParameter(
                    name="denominator",
                    description=(
                        "'retrieved' (default) divides by the number of distinct "
                        "results returned; 'k' divides by the cut-off."
                    ),
                    default="retrieved",
                ),
            ),
        ),
        EvaluatorSpecification(
            type="reciprocal_rank",
            kind=ReciprocalRankEvaluator.kind,
            description=ReciprocalRankEvaluator.description,
            factory=_build_reciprocal_rank,
            parameters=(
                EvaluatorParameter(
                    name="k",
                    description="Optional cut-off rank; omit to search the whole list.",
                ),
            ),
        ),
        EvaluatorSpecification(
            type="citation_f1",
            kind=CitationF1Evaluator.kind,
            description=CitationF1Evaluator.description,
            factory=_build_citation_f1,
        ),
        EvaluatorSpecification(
            type="citation_validity",
            kind=CitationValidityEvaluator.kind,
            description=CitationValidityEvaluator.description,
            factory=_build_citation_validity,
        ),
        EvaluatorSpecification(
            type="answer_exact_match",
            kind=AnswerExactMatchEvaluator.kind,
            description=AnswerExactMatchEvaluator.description,
            factory=_build_answer_exact_match,
        ),
        EvaluatorSpecification(
            type="answer_token_f1",
            kind=AnswerTokenF1Evaluator.kind,
            description=AnswerTokenF1Evaluator.description,
            factory=_build_answer_token_f1,
        ),
        EvaluatorSpecification(
            type="lexical_groundedness",
            kind=LexicalGroundednessEvaluator.kind,
            description=LexicalGroundednessEvaluator.description,
            factory=_build_lexical_groundedness,
        ),
        EvaluatorSpecification(
            type="llm_judge_groundedness",
            kind=EvaluatorKind.LLM_JUDGE,
            description=(
                "Asks a judge model whether the answer is supported by the retrieved "
                "context. Requires a judge provider; results carry evaluator uncertainty."
            ),
            factory=_build_judge_groundedness,
            requires_judge=True,
        ),
        EvaluatorSpecification(
            type="llm_judge_answer_relevance",
            kind=EvaluatorKind.LLM_JUDGE,
            description=(
                "Asks a judge model whether the answer addresses the question. "
                "Requires a judge provider; results carry evaluator uncertainty."
            ),
            factory=_build_judge_answer_relevance,
            requires_judge=True,
        ),
    )
}


def available_evaluator_types() -> tuple[str, ...]:
    """Every registered evaluator type, sorted for stable output."""
    return tuple(sorted(_REGISTRY))


def iter_specifications() -> tuple[EvaluatorSpecification, ...]:
    """Every registered specification, sorted by type."""
    return tuple(_REGISTRY[name] for name in available_evaluator_types())


def get_specification(evaluator_type: str) -> EvaluatorSpecification:
    """Look up one specification.

    Args:
        evaluator_type: The registry key.

    Returns:
        The specification.

    Raises:
        UnknownEvaluatorError: If the type is not registered.
    """
    try:
        return _REGISTRY[evaluator_type]
    except KeyError:
        raise UnknownEvaluatorError(evaluator_type, available_evaluator_types()) from None


def _validate_parameters(specification: EvaluatorSpecification, params: Mapping[str, Any]) -> None:
    """Reject unknown and missing parameters before building."""
    known = {parameter.name for parameter in specification.parameters}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ConfigError(
            f"Evaluator {specification.type!r} received unknown parameter(s): "
            f"{', '.join(unknown)}. Accepted parameter(s): "
            f"{', '.join(sorted(known)) or '(none)'}"
        )
    missing = sorted(
        parameter.name
        for parameter in specification.parameters
        if parameter.required and parameter.name not in params
    )
    if missing:
        raise ConfigError(
            f"Evaluator {specification.type!r} is missing required parameter(s): "
            f"{', '.join(missing)}"
        )


def build_evaluator(
    evaluator_type: str,
    params: Mapping[str, Any] | None = None,
    min_case_score: float | None = None,
    context: EvaluatorBuildContext | None = None,
) -> Evaluator:
    """Build an evaluator instance from configuration values.

    Args:
        evaluator_type: A registered type, e.g. ``"recall_at_k"``.
        params: Type-specific parameters.
        min_case_score: Optional per-case threshold.
        context: Shared build resources; defaults to an empty context with no
            judge provider.

    Returns:
        The configured evaluator.

    Raises:
        UnknownEvaluatorError: If ``evaluator_type`` is not registered.
        ConfigError: If parameters are unknown, missing, of the wrong type, or
            outside a valid range, or if a judge is required but not configured.
    """
    specification = get_specification(evaluator_type)
    effective_params = dict(params or {})
    _validate_parameters(specification, effective_params)
    effective_context = context or EvaluatorBuildContext()

    try:
        return specification.factory(effective_params, min_case_score, effective_context)
    except ValueError as exc:
        # Constructor-level validation (k >= 1, threshold range) surfaces as a
        # configuration error so the CLI can exit with the usage code.
        raise ConfigError(f"Evaluator {evaluator_type!r} is misconfigured: {exc}") from exc
