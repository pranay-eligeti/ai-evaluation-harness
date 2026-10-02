"""AI Evaluation Harness: a deterministic-first evaluation framework for RAG and LLM systems.

The package is organised in layers, each of which can be used independently:

``models``
    The typed evaluation data model (cases, retrieved documents, datasets).
``metrics``
    Pure functions implementing the deterministic metrics. No I/O, no state.
``evaluators``
    The evaluator contract; adapts metrics to cases and produces structured results.
``judges``
    A provider-neutral abstraction for LLM-as-a-judge evaluation.
``datasets``
    Dataset loading and validation.
``runner``
    Suite configuration, execution, aggregation, and quality gates.
``reporting``
    Structured JSON reports and human-readable console output.
``cli``
    The ``ai-eval`` command-line entry point.

Importing this package never performs network access.
"""

from ai_eval_harness.models import EvaluationCase, EvaluationDataset, RetrievedDocument
from ai_eval_harness.results import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorKind,
    Measurement,
)

__version__ = "0.3.0"

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluatorKind",
    "Measurement",
    "RetrievedDocument",
    "__version__",
]
