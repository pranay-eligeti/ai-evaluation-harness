"""Execute evaluators, retain failures, aggregate separately, and apply gates."""

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ai_eval_harness import __version__
from ai_eval_harness.errors import ConfigError, EmptyDatasetError
from ai_eval_harness.evaluators.base import Evaluator
from ai_eval_harness.evaluators.registry import EvaluatorBuildContext, build_evaluator
from ai_eval_harness.metrics.aggregation import mean
from ai_eval_harness.models import EvaluationDataset
from ai_eval_harness.results import EvaluationResult, EvaluationStatus
from ai_eval_harness.runner.config import SuiteConfig


class Aggregate(BaseModel):
    mean_score: float | None
    pass_rate: float | None
    scored_count: int
    skipped_count: int
    error_count: int
    statuses: dict[str, int]


class GateResult(BaseModel):
    evaluator: str
    passed: bool
    explanation: str


class SuiteReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1"
    harness_version: str = __version__
    dataset: dict[str, Any]
    configuration: SuiteConfig
    evaluators: list[dict[str, str]]
    results: list[EvaluationResult]
    aggregates: dict[str, Aggregate]
    gates: list[GateResult]
    status: Literal["passed", "failed", "error", "insufficient_data"]


def aggregate(results: list[EvaluationResult]) -> Aggregate:
    counts = Counter(result.status.value for result in results)
    scores = [result.score for result in results if result.score is not None]
    verdicts = counts["passed"] + counts["failed"]
    return Aggregate(
        mean_score=mean(scores),
        pass_rate=counts["passed"] / verdicts if verdicts else None,
        scored_count=len(scores),
        skipped_count=counts["skipped"],
        error_count=counts["error"],
        statuses=dict(sorted(counts.items())),
    )


def run_suite(
    dataset: EvaluationDataset,
    config: SuiteConfig,
    *,
    context: EvaluatorBuildContext | None = None,
    evaluators: list[Evaluator] | None = None,
) -> SuiteReport:
    if not dataset.cases and not config.allow_empty_dataset:
        raise EmptyDatasetError(dataset.name)
    instances = (
        evaluators
        if evaluators is not None
        else [
            build_evaluator(item.type, item.params, item.min_case_score, context)
            for item in config.evaluators
        ]
    )
    names = [instance.name for instance in instances]
    if not names or len(set(names)) != len(names):
        raise ConfigError("Suite must contain evaluators with unique instance names")
    if any(gate.evaluator not in names for gate in config.gates):
        raise ConfigError("Quality gate references an evaluator absent from this suite")
    results: list[EvaluationResult] = []
    for case in dataset.cases:
        for instance in instances:
            try:
                result = instance.evaluate(case)
                if (result.case_id, result.evaluator, result.kind, result.evaluator_type) != (
                    case.case_id,
                    instance.name,
                    instance.kind,
                    instance.evaluator_type,
                ):
                    raise ValueError("Evaluator returned inconsistent result identity")
                results.append(result)
            except Exception as exc:
                results.append(
                    EvaluationResult(
                        evaluator=instance.name,
                        evaluator_type=instance.evaluator_type,
                        kind=instance.kind,
                        case_id=case.case_id,
                        status=EvaluationStatus.ERROR,
                        explanation=f"{type(exc).__name__}: {exc}",
                    )
                )
    aggregates = {
        name: aggregate([result for result in results if result.evaluator == name])
        for name in names
    }
    gates: list[GateResult] = []
    for gate in config.gates:
        values = aggregates[gate.evaluator]
        failures: list[str] = []
        for field, threshold, minimum in (
            ("mean_score", gate.min_mean_score, True),
            ("pass_rate", gate.min_pass_rate, True),
            ("scored_count", gate.min_scored_count, True),
            ("skipped_count", gate.max_skipped_count, False),
            ("error_count", gate.max_error_count, False),
        ):
            if threshold is None:
                continue
            actual = getattr(values, field)
            if actual is None or (actual < threshold if minimum else actual > threshold):
                failures.append(
                    f"{field}={actual}, required {'>=' if minimum else '<='}{threshold}"
                )
        gates.append(
            GateResult(
                evaluator=gate.evaluator,
                passed=not failures,
                explanation="; ".join(failures)
                if failures
                else "All configured thresholds satisfied",
            )
        )
    status: Literal["passed", "failed", "error", "insufficient_data"] = "passed"
    if any(result.status == EvaluationStatus.ERROR for result in results):
        status = "error"
    elif any(not gate.passed for gate in gates):
        status = "failed"
    elif not results or not any(result.score is not None for result in results):
        status = "insufficient_data"
    return SuiteReport(
        dataset={
            "name": dataset.name,
            "case_count": len(dataset),
            "checksum": dataset.checksum,
            "source_path": dataset.source_path,
            "case_metadata": {case.case_id: case.metadata for case in dataset.cases},
        },
        configuration=config,
        evaluators=[
            {"name": item.name, "type": item.evaluator_type, "kind": item.kind.value}
            for item in instances
        ],
        results=results,
        aggregates=aggregates,
        gates=gates,
        status=status,
    )
