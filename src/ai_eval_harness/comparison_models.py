"""Independent comparison schema and regression configuration."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_eval_harness import __version__
from ai_eval_harness.runner.suite import GateResult

Change = Literal["improved", "unchanged", "regressed", "not_comparable"]


class ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RegressionGate(ComparisonModel):
    evaluator: str = Field(min_length=1)
    metric: Literal["mean_score", "pass_rate"] = "mean_score"
    max_drop: float = Field(ge=0, le=1, strict=True)


class ComparisonConfig(ComparisonModel):
    exact_population: bool = Field(default=True, strict=True)
    regression_gates: list[RegressionGate] = Field(default_factory=list)


class Delta(ComparisonModel):
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    status: Change = "not_comparable"
    reason: str = ""


class CaseChange(ComparisonModel):
    case_id: str
    population_status: Literal["matched", "added", "removed"]
    metrics: dict[str, Delta]
    baseline_results: dict[str, Any]
    candidate_results: dict[str, Any]


class MetricComparison(ComparisonModel):
    compatible: bool
    reason: str
    mean_score: Delta
    pass_rate: Delta


class RegressionResult(ComparisonModel):
    evaluator: str
    metric: str
    max_drop: float
    passed: bool
    explanation: str


class Population(ComparisonModel):
    baseline_count: int
    candidate_count: int
    matched_count: int
    added: list[str]
    removed: list[str]
    semantics: Literal["exact", "matched_only"]


class ComparisonReport(ComparisonModel):
    schema_version: Literal["comparison-1"] = "comparison-1"
    harness_version: str = __version__
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    configuration: ComparisonConfig
    compatible: bool
    population: Population
    evaluators: dict[str, MetricComparison]
    cases: list[CaseChange]
    regression_gates: list[RegressionResult]
    candidate_absolute_gates: list[GateResult]
    status: Literal["passed", "failed", "error"]
    provenance: dict[str, str]


def load_comparison_config(path: Path) -> ComparisonConfig:
    return ComparisonConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
