"""Versioned application-neutral captures; list position is retrieval rank."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_eval_harness.models import EvaluationCase, EvaluationDataset


class CapturedRun(BaseModel):
    """Capture v1 reuses the strict case contract without changing report schemas.

    Document IDs identify the evaluation unit (source or chunk). Source IDs,
    chunk offsets and application-specific details belong in document metadata.
    Explicit rank fields are forbidden: rank is the ordered list position.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal["capture-1"] = "capture-1"
    run_id: str = Field(min_length=1)
    system: str | None = None
    revision: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_capture(self) -> "CapturedRun":
        if not self.run_id.strip():
            raise ValueError("run_id must not be blank")
        json.dumps(self.model_dump(), allow_nan=False)
        for case in self.cases:
            for document in case.retrieved_documents:
                for key in ("source_id", "chunk_id"):
                    value = document.metadata.get(key)
                    if key in document.metadata and (
                        not isinstance(value, str) or not value.strip()
                    ):
                        raise ValueError(f"Document metadata {key} must be nonblank text")
        EvaluationDataset(name=self.run_id, cases=self.cases)
        return self

    def dataset(self, path: Path | None = None) -> EvaluationDataset:
        return EvaluationDataset(
            name=self.run_id,
            cases=self.cases,
            source_path=str(path) if path else None,
            checksum=hashlib.sha256(self.model_dump_json().encode()).hexdigest(),
        )

    def identification(self) -> dict[str, Any]:
        return self.model_dump(exclude={"cases"}, mode="json")


def load_capture(path: Path) -> CapturedRun:
    return CapturedRun.model_validate_json(path.read_text(encoding="utf-8"))


def case_identity(case: EvaluationCase) -> str:
    """Match question and annotations, excluding system outputs and labels."""
    payload = case.model_dump(
        include={"question", "reference_answer", "expected_document_ids", "expected_citations"}
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
