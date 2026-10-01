import json
from typing import Any

import pytest
from pydantic import ValidationError

from ai_eval_harness.datasets import load_dataset_from_records
from ai_eval_harness.errors import (
    ConfigError,
    DatasetValidationError,
    EmptyDatasetError,
    JudgeResponseError,
)
from ai_eval_harness.evaluators import build_evaluator
from ai_eval_harness.evaluators.registry import EvaluatorBuildContext
from ai_eval_harness.judges import ScriptedJudgeProvider, parse_verdict
from ai_eval_harness.models import (
    EvaluationCase,
    EvaluationDataset,
    RetrievedDocument,
    rank_documents,
)
from ai_eval_harness.results import EvaluationResult, EvaluationStatus, Measurement
from ai_eval_harness.runner import QualityGate


@pytest.mark.parametrize(
    "payload",
    [
        {"case_id": "", "question": "q"},
        {"case_id": "a", "question": " "},
        {"case_id": "a", "question": "q", "unknown": 1},
        {"case_id": "a", "question": "q", "expected_document_ids": [" "]},
        {"case_id": "a", "question": "q", "actual_citations": [2]},
        {
            "case_id": "a",
            "question": "q",
            "retrieved_documents": [{"doc_id": "a", "score": float("nan")}],
        },
    ],
)
def test_invalid_models(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_duplicate_cases_and_empty() -> None:
    case = EvaluationCase(case_id="a", question="q")
    with pytest.raises(ValidationError):
        EvaluationDataset(name="d", cases=[case, case])
    with pytest.raises(EmptyDatasetError):
        load_dataset_from_records([])
    with pytest.raises(DatasetValidationError) as error:
        load_dataset_from_records([{"case_id": "a"}, {"question": "q"}])
    assert len(error.value.problems) == 2


def test_explicit_ranking() -> None:
    docs = [
        RetrievedDocument(doc_id="u"),
        RetrievedDocument(doc_id="a", score=2),
        RetrievedDocument(doc_id="b", score=2),
        RetrievedDocument(doc_id="c", score=3),
    ]
    assert [doc.doc_id for doc in rank_documents(docs)] == ["c", "a", "b", "u"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_bad_gate(threshold: Any) -> None:
    with pytest.raises(ValidationError):
        QualityGate(evaluator="a", min_mean_score=threshold)


@pytest.mark.parametrize("score", [-1, 2, float("nan"), float("inf")])
def test_bad_measurement(score: float) -> None:
    with pytest.raises(ValueError):
        Measurement(score, "reason")


@pytest.mark.parametrize(
    "name",
    [
        "recall_at_k",
        "precision_at_k",
        "reciprocal_rank",
        "citation_f1",
        "answer_exact_match",
        "answer_token_f1",
    ],
)
def test_missing_annotations_skip(name: str) -> None:
    params = {"k": 3} if name.endswith("at_k") else {}
    result = build_evaluator(name, params).evaluate(EvaluationCase(case_id="a", question="q"))
    assert result.status == EvaluationStatus.SKIPPED
    assert result.score is None


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("unknown", {}),
        ("recall_at_k", {}),
        ("recall_at_k", {"k": True}),
        ("answer_token_f1", {"k": 1}),
        ("llm_judge_groundedness", {}),
    ],
)
def test_registry_errors(name: str, params: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        build_evaluator(name, params)


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"score":2,"reasoning":"x"}',
        '{"score":NaN,"reasoning":"x"}',
        '{"score":"1","reasoning":"x"}',
        '{"score":1,"reasoning":" "}',
    ],
)
def test_judge_parse_errors(payload: str) -> None:
    with pytest.raises(JudgeResponseError):
        parse_verdict(payload)


def test_offline_judge_pipeline() -> None:
    provider = ScriptedJudgeProvider(
        {"groundedness": {"a": {"score": 0.75, "reasoning": "partial"}}}
    )
    evaluator = build_evaluator(
        "llm_judge_groundedness", min_case_score=0.8, context=EvaluatorBuildContext(provider)
    )
    case = EvaluationCase(
        case_id="a",
        question="q",
        generated_answer="blue",
        retrieved_documents=[RetrievedDocument(doc_id="s", text="blue")],
    )
    result = evaluator.evaluate(case)
    assert result.status == EvaluationStatus.FAILED
    assert result.score == 0.75
    assert result.metadata["judge_model"].startswith("scripted:")
    assert result.metadata["prompt_version"] == "1"
    missing = case.model_copy(update={"case_id": "unrecorded"})
    assert evaluator.evaluate(missing).status == EvaluationStatus.ERROR
    assert (
        parse_verdict("```json\n" + json.dumps({"score": True, "reasoning": "yes"}) + "\n```").score
        == 1
    )


@pytest.mark.parametrize(
    ("status", "score", "threshold"),
    [
        ("scored", None, None),
        ("skipped", 1, None),
        ("error", 0, None),
        ("passed", 0.2, 0.5),
        ("failed", 0.8, 0.5),
        ("passed", 1, None),
        ("scored", 0.5, 0.5),
        ("scored", float("nan"), None),
    ],
)
def test_inconsistent_results(status: str, score: float | None, threshold: float | None) -> None:
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            {
                "evaluator": "test",
                "evaluator_type": "test",
                "kind": "deterministic",
                "case_id": "a",
                "status": status,
                "score": score,
                "threshold": threshold,
                "explanation": "reason",
            }
        )
