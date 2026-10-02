"""Regression workflow boundaries, including historical report compatibility."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai_eval_harness.capture import CapturedRun, load_capture
from ai_eval_harness.cli.main import main
from ai_eval_harness.comparison import ComparisonError, compare_reports, validate_report
from ai_eval_harness.comparison_models import ComparisonConfig, ComparisonReport, RegressionGate
from ai_eval_harness.evaluators.registry import EvaluatorBuildContext
from ai_eval_harness.judges.scripted import ScriptedJudgeProvider
from ai_eval_harness.models import EvaluationCase, EvaluationDataset, RetrievedDocument
from ai_eval_harness.reporting import report_json
from ai_eval_harness.runner.config import EvaluatorConfig, QualityGate, SuiteConfig
from ai_eval_harness.runner.suite import SuiteReport, run_suite


def suite(
    ids: tuple[str, ...] = ("a", "b", "c"),
    hits: tuple[bool, ...] = (True, False, True),
    *,
    k: int = 1,
    threshold: float | None = 0.5,
    gate: float | None = None,
) -> SuiteReport:
    cases = [
        EvaluationCase(
            case_id=i,
            question="Question " + i,
            expected_document_ids=["gold"],
            retrieved_documents=[RetrievedDocument(doc_id="gold" if hit else "noise")],
        )
        for i, hit in zip(ids, hits, strict=True)
    ]
    config = SuiteConfig(
        evaluators=[
            EvaluatorConfig(
                type="recall_at_k",
                params={"k": k},
                min_case_score=threshold,
            )
        ],
        gates=[QualityGate(evaluator=f"recall@{k}", min_mean_score=gate)] if gate else [],
    )
    return run_suite(EvaluationDataset(name="test", cases=cases), config)


def settings(
    drop: float = 0.0, *, exact: bool = True, metric: str = "mean_score"
) -> ComparisonConfig:
    return ComparisonConfig.model_validate(
        {
            "exact_population": exact,
            "regression_gates": [{"evaluator": "recall@1", "metric": metric, "max_drop": drop}],
        }
    )


def test_identical_stable_json_and_old_reports() -> None:
    report = suite()
    result = compare_reports(report, report, settings())
    assert result.status == "passed"
    assert all(c.metrics["recall@1"].status == "unchanged" for c in result.cases)
    assert ComparisonReport.model_validate_json(result.model_dump_json()) == result
    assert result.model_dump_json() == compare_reports(report, report, settings()).model_dump_json()
    for version in ("1", "2"):
        payload = report.model_dump(mode="json")
        payload["schema_version"] = version
        payload["harness_version"] = "0.2.0"
        payload["dataset"].pop("case_identity")
        payload["dataset"]["checksum"] = "historical-identical-bytes"
        old = SuiteReport.model_validate(payload)
        assert compare_reports(old, old, settings()).status == "passed"


@pytest.mark.parametrize(
    "hits,expected",
    [
        ((True, True, True), "improved"),
        ((False, False, False), "regressed"),
        ((True, False, True), "unchanged"),
    ],
)
def test_metric_changes(hits: tuple[bool, ...], expected: str) -> None:
    result = compare_reports(suite(), suite(hits=hits), settings(1.0))
    assert result.evaluators["recall@1"].mean_score.status == expected


def test_case_changes_and_aggregate_cancel() -> None:
    result = compare_reports(suite(), suite(hits=(False, True, True)), settings())
    assert result.evaluators["recall@1"].mean_score.delta == 0
    assert [c.metrics["recall@1"].status for c in result.cases] == [
        "regressed",
        "improved",
        "unchanged",
    ]
    assert result.cases[0].baseline_results["recall@1"]["status"] == "passed"
    assert result.cases[0].candidate_results["recall@1"]["status"] == "failed"


@pytest.mark.parametrize("drop,passed", [(0.0, False), (0.3333333333333333, True), (1.0, True)])
def test_regression_tolerance(drop: float, passed: bool) -> None:
    result = compare_reports(suite(), suite(hits=(False, False, True)), settings(drop))
    assert result.regression_gates[0].passed is passed
    assert result.status == ("passed" if passed else "failed")


def test_absolute_and_relative_independent() -> None:
    # Meets absolute threshold but regresses; improves but misses absolute threshold.
    regressed = compare_reports(suite(), suite(hits=(False, False, True), gate=0.3), settings())
    assert regressed.candidate_absolute_gates[0].passed and regressed.status == "failed"
    improved = compare_reports(suite(hits=(False, False, False)), suite(gate=0.9), settings())
    assert improved.regression_gates[0].passed
    assert not improved.candidate_absolute_gates[0].passed and improved.status == "failed"


@pytest.mark.parametrize("drop", [-0.1, 1.1, float("nan"), float("inf"), "0.1", True])
def test_bad_gate(drop: Any) -> None:
    with pytest.raises(ValidationError):
        RegressionGate(evaluator="recall@1", max_drop=drop)


def test_pass_rate_gate_and_missing_verdict() -> None:
    assert compare_reports(suite(), suite(), settings(metric="pass_rate")).status == "passed"
    result = compare_reports(
        suite(threshold=None), suite(threshold=None), settings(metric="pass_rate")
    )
    assert not result.regression_gates[0].passed and result.status == "failed"


def test_population_and_partial_overlap() -> None:
    candidate = suite(ids=("b", "c", "d"), hits=(False, True, True))
    exact = compare_reports(suite(), candidate, settings())
    assert exact.status == "error" and exact.evaluators["recall@1"].mean_score.delta is None
    partial = compare_reports(suite(), candidate, settings(exact=False))
    assert partial.status == "passed" and partial.population.matched_count == 2
    assert partial.population.added == ["d"] and partial.population.removed == ["a"]
    assert [c.population_status for c in partial.cases] == [
        "removed",
        "matched",
        "matched",
        "added",
    ]
    assert partial.evaluators["recall@1"].mean_score.baseline == 0.5


def test_no_overlap_and_unknown_gate() -> None:
    with pytest.raises(ComparisonError, match="overlapping"):
        compare_reports(suite(), suite(ids=("d", "e", "f")), settings())
    with pytest.raises(ValueError, match="absent"):
        compare_reports(
            suite(),
            suite(),
            ComparisonConfig(regression_gates=[RegressionGate(evaluator="absent", max_drop=0.0)]),
        )


def test_missing_metric_and_cutoff_incompatible() -> None:
    result = compare_reports(suite(), suite(k=3), settings())
    assert result.status == "error"
    assert set(result.evaluators) == {"recall@1", "recall@3"}
    assert not result.regression_gates[0].passed
    result = compare_reports(suite(), suite(threshold=0.9), settings())
    assert result.evaluators["recall@1"].reason == "Evaluator configuration mismatch"


def test_precision_denominator_and_default_normalization() -> None:
    data = EvaluationDataset(
        name="precision",
        cases=[
            EvaluationCase(
                case_id="a",
                question="q",
                expected_document_ids=["gold"],
                retrieved_documents=[RetrievedDocument(doc_id="gold")],
            )
        ],
    )

    def run(denominator: str | None) -> SuiteReport:
        params: dict[str, Any] = {"k": 3}
        if denominator:
            params["denominator"] = denominator
        return run_suite(
            data, SuiteConfig(evaluators=[EvaluatorConfig(type="precision_at_k", params=params)])
        )

    # Names differ for denominator semantics; no magic evaluator matching.
    assert compare_reports(run(None), run("retrieved"), ComparisonConfig()).status == "passed"
    assert compare_reports(run(None), run("k"), ComparisonConfig()).status == "error"


def test_question_identity_and_coverage_loss() -> None:
    candidate = suite()
    candidate.dataset["case_identity"]["a"] = "unrelated-question"
    assert compare_reports(suite(), candidate, settings()).status == "error"
    dataset = EvaluationDataset(
        name="missing",
        cases=[
            EvaluationCase(
                case_id=i,
                question="Question " + i,
                expected_document_ids=None if i == "a" else ["gold"],
                retrieved_documents=[RetrievedDocument(doc_id="gold")],
            )
            for i in ("a", "b", "c")
        ],
    )
    missing = run_suite(dataset, suite().configuration)
    # Preserve annotation identity to isolate the score-coverage rule.
    missing.dataset["case_identity"] = suite().dataset["case_identity"]
    assert "coverage" in compare_reports(suite(), missing, settings()).evaluators["recall@1"].reason


@pytest.mark.parametrize(
    "field", ["prompt_version", "judge_provider", "configured_model", "judge_model", "criterion"]
)
def test_semantic_provenance(field: str) -> None:
    data = EvaluationDataset(
        name="judge", cases=[EvaluationCase(case_id="a", question="q", generated_answer="a")]
    )
    provider = ScriptedJudgeProvider({"answer_relevance": {"a": {"score": 0.8, "reasoning": "ok"}}})
    config = SuiteConfig(evaluators=[EvaluatorConfig(type="llm_judge_answer_relevance")])
    report = run_suite(data, config, context=EvaluatorBuildContext(provider))
    assert compare_reports(report, report, ComparisonConfig()).status == "passed"
    candidate = report.model_copy(deep=True)
    candidate.results[0].metadata[field] = "changed"
    assert compare_reports(report, candidate, ComparisonConfig()).status == "error"


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing_aggregate", "wrong_aggregate", "missing_result", "schema"]
)
def test_invalid_report(mutation: str) -> None:
    payload = suite().model_dump(mode="json")
    if mutation == "duplicate":
        payload["results"].append(payload["results"][0])
    elif mutation == "missing_aggregate":
        payload["aggregates"].clear()
    elif mutation == "wrong_aggregate":
        payload["aggregates"]["recall@1"]["mean_score"] = 0.123
    elif mutation == "missing_result":
        payload["results"].pop()
    else:
        payload["schema_version"] = "99"
    with pytest.raises(ValueError):
        validate_report(SuiteReport.model_validate(payload))


@pytest.mark.parametrize(
    "mutation", ["duplicate", "empty", "blank", "rank", "source", "score", "extra", "metadata_nan"]
)
def test_capture_invalid(mutation: str) -> None:
    case = {"case_id": "a", "question": "q", "retrieved_documents": [{"doc_id": "source"}]}
    payload: dict[str, Any] = {"run_id": "run", "cases": [case]}
    if mutation == "duplicate":
        payload["cases"].append(case)
    elif mutation == "empty":
        payload["cases"] = []
    elif mutation == "blank":
        payload["run_id"] = " "
    elif mutation == "rank":
        case["retrieved_documents"][0]["rank"] = 0  # type: ignore[index]
    elif mutation == "source":
        case["retrieved_documents"][0]["metadata"] = {"source_id": []}  # type: ignore[index]
    elif mutation == "score":
        case["retrieved_documents"][0]["score"] = float("inf")  # type: ignore[index]
    elif mutation == "extra":
        payload["schema_version"] = "wrong"
    else:
        payload["metadata"] = {"bad": float("nan")}
    with pytest.raises(ValueError):
        CapturedRun.model_validate(payload)


def test_optional_capture() -> None:
    capture = CapturedRun(run_id="minimal", cases=[EvaluationCase(case_id="a", question="q")])
    assert capture.system is None and capture.dataset().cases[0].reference_answer is None


@pytest.mark.parametrize(
    "scenario,exit_code",
    [
        ("pass", 0),
        ("fail", 1),
        ("bad_config", 2),
        ("bad_report", 2),
        ("no_overlap", 3),
        ("bad_gate", 2),
        ("incompatible", 3),
    ],
)
def test_cli(tmp_path: Path, scenario: str, exit_code: int) -> None:
    baseline = suite()
    candidate = suite(hits=(False, False, True)) if scenario == "fail" else suite()
    if scenario == "no_overlap":
        candidate = suite(ids=("d", "e", "f"))
    if scenario == "incompatible":
        candidate = suite(k=3)
    (tmp_path / "left.json").write_text(report_json(baseline))
    (tmp_path / "right.json").write_text(report_json(candidate))
    config = '[[regression_gates]]\nevaluator="recall@1"\nmax_drop=0.0\n'
    if scenario == "bad_config":
        config += "unknown=1\n"
    if scenario == "bad_gate":
        config = config.replace("recall@1", "absent")
    (tmp_path / "config.toml").write_text(config)
    if scenario == "bad_report":
        (tmp_path / "left.json").write_text("{}")
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(tmp_path / "left.json"),
                "--candidate",
                str(tmp_path / "right.json"),
                "--config",
                str(tmp_path / "config.toml"),
                "--report",
                str(tmp_path / "comparison.json"),
            ]
        )
        == exit_code
    )
    if scenario in ("pass", "fail", "incompatible"):
        ComparisonReport.model_validate_json((tmp_path / "comparison.json").read_text())


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--help"])
    assert exc.value.code == 0


def test_semantic_regression_gate() -> None:
    data = EvaluationDataset(
        name="semantic",
        cases=[
            EvaluationCase(
                case_id="a",
                question="q",
                generated_answer="answer",
                retrieved_documents=[RetrievedDocument(doc_id="source", text="answer")],
            )
        ],
    )
    config = SuiteConfig(evaluators=[EvaluatorConfig(type="llm_judge_groundedness")])

    def run(score: float) -> SuiteReport:
        provider = ScriptedJudgeProvider(
            {"groundedness": {"a": {"score": score, "reasoning": "offline fixture"}}}
        )
        return run_suite(data, config, context=EvaluatorBuildContext(provider))

    gate = ComparisonConfig(
        regression_gates=[RegressionGate(evaluator="llm_judge_groundedness", max_drop=0.05)]
    )
    baseline = run(0.8)
    assert compare_reports(baseline, run(0.9), gate).status == "passed"
    assert compare_reports(baseline, run(0.75), gate).status == "passed"
    failed = compare_reports(baseline, run(0.6), gate)
    assert failed.status == "failed"
    assert failed.cases[0].metrics["llm_judge_groundedness"].status == "regressed"


def test_execution_write_error(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(report_json(suite()))
    (tmp_path / "config.toml").write_text("")
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(tmp_path / "report.json"),
                "--candidate",
                str(tmp_path / "report.json"),
                "--config",
                str(tmp_path / "config.toml"),
                "--report",
                str(tmp_path),
            ]
        )
        == 3
    )


def test_no_comparable_scores_and_metric_type() -> None:
    data = EvaluationDataset(name="no-labels", cases=[EvaluationCase(case_id="a", question="q")])
    empty = run_suite(data, SuiteConfig(evaluators=[EvaluatorConfig(type="answer_exact_match")]))
    assert compare_reports(empty, empty, ComparisonConfig()).status == "error"
    other = run_suite(data, SuiteConfig(evaluators=[EvaluatorConfig(type="answer_token_f1")]))
    assert compare_reports(empty, other, ComparisonConfig()).status == "error"


def test_annotation_and_forged_gate_rejected() -> None:
    left, right = suite(), suite()
    right.dataset["case_identity"]["a"] = "changed annotation"
    assert compare_reports(left, right, settings()).status == "error"
    bad = suite(gate=0.9)
    bad.gates[0].passed = True
    with pytest.raises(ValueError, match="gate results"):
        validate_report(bad)


def test_capture_fixture_workflow(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("baseline", "candidate"):
        capture = load_capture(root / f"examples/rag/{name}.json")
        assert capture.system == "rag-knowledge-assistant/tfidf-extractive"
        assert (
            main(
                [
                    "run",
                    "--capture",
                    str(root / f"examples/rag/{name}.json"),
                    "--config",
                    str(root / "examples/rag/suite.toml"),
                    "--report",
                    str(tmp_path / f"{name}.json"),
                ]
            )
            == 0
        )
    args = [
        "compare",
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--candidate",
        str(tmp_path / "candidate.json"),
        "--report",
        str(tmp_path / "compare.json"),
    ]
    assert main([*args, "--config", str(root / "examples/rag/regression-pass.toml")]) == 0
    assert main([*args, "--config", str(root / "examples/rag/regression-fail.toml")]) == 1
