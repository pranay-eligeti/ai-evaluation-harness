"""Strict suite configuration, shared by the CLI and Python callers."""

import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from ai_eval_harness.errors import ConfigError

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, strict=True)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class EvaluatorConfig(ConfigModel):
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    min_case_score: Score | None = None


class QualityGate(ConfigModel):
    evaluator: str = Field(min_length=1)
    min_mean_score: Score | None = None
    min_pass_rate: Score | None = None
    min_scored_count: Count = 1
    max_skipped_count: Count | None = None
    max_error_count: Count = 0


class SuiteConfig(ConfigModel):
    evaluators: list[EvaluatorConfig] = Field(min_length=1)
    gates: list[QualityGate] = Field(default_factory=list)
    allow_empty_dataset: bool = False
    judge_responses: str | None = None


def load_config(path: Path) -> SuiteConfig:
    try:
        return SuiteConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Could not load configuration {path}: {exc}") from exc
