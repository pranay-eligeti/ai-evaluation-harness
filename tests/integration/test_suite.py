from pathlib import Path

import pytest

from ai_eval_harness.datasets import load_dataset
from ai_eval_harness.errors import ConfigError, DatasetValidationError
from ai_eval_harness.evaluators import BaseEvaluator
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.reporting import report_json
from ai_eval_harness.results import EvaluatorKind, Measurement
from ai_eval_harness.runner import QualityGate, SuiteReport, run_suite
from ai_eval_harness.runner.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_sample_and_deterministic_report() -> None:
    dataset = load_dataset(ROOT / "examples/datasets/synthetic.json")
    config = load_config(ROOT / "examples/suite.toml")
    report = run_suite(dataset, config)
    assert report.status == "passed"
    assert len(report.results) == 40
    assert report.aggregates["recall@3"].mean_score == 0.5
    assert report.aggregates["reciprocal_rank"].mean_score == 0.375
    assert report.aggregates["citation_f1"].mean_score == 0.6
    assert report.aggregates["recall@3"].skipped_count == 1
    assert SuiteReport.model_validate_json(report_json(report)) == report
    assert report_json(report) == report_json(run_suite(dataset, config))


def test_gates_fail_without_hiding_results() -> None:
    report = run_suite(
        load_dataset(ROOT / "examples/datasets/synthetic.json"),
        load_config(ROOT / "examples/failing.toml"),
    )
    assert report.status == "failed"
    assert len(report.results) == 40
    assert not report.gates[0].passed
    assert "required >=0.9" in report.gates[0].explanation


@pytest.mark.parametrize(
    "text",
    [
        '[{"case_id":"a"}]',
        "[1]",
        '{"cases":{}}',
        '[{"case_id":"a","question":"q"},{"case_id":"a","question":"q"}]',
    ],
)
def test_bad_dataset_files(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_jsonl_errors_include_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n\nno\n")
    with pytest.raises(DatasetValidationError) as error:
        load_dataset(path)
    assert "line 1" in str(error.value) and "line 3" in str(error.value)


def test_empty_and_unscored_cannot_pass(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]")
    config = load_config(ROOT / "examples/suite.toml")
    config.allow_empty_dataset = True
    report = run_suite(load_dataset(path, allow_empty=True), config)
    assert report.status == "failed"
    config.gates = []
    assert run_suite(load_dataset(path, allow_empty=True), config).status == "insufficient_data"


def test_unknown_and_duplicate_evaluators() -> None:
    dataset = load_dataset(ROOT / "examples/datasets/synthetic.json")
    config = load_config(ROOT / "examples/suite.toml")
    config.gates = [QualityGate(evaluator="unknown")]
    with pytest.raises(ConfigError):
        run_suite(dataset, config)
    config.gates = []
    config.evaluators.append(config.evaluators[0])
    with pytest.raises(ConfigError):
        run_suite(dataset, config)


class BrokenEvaluator(BaseEvaluator):
    _evaluator_type = "broken"
    kind = EvaluatorKind.DETERMINISTIC
    description = "Test failure capture"

    def measure(self, case: EvaluationCase) -> Measurement:
        raise RuntimeError(f"failed {case.case_id}")


def test_evaluator_errors_fail_suite() -> None:
    dataset = load_dataset(ROOT / "examples/datasets/synthetic.json")
    config = load_config(ROOT / "examples/suite.toml")
    config.gates = []
    report = run_suite(dataset, config, evaluators=[BrokenEvaluator()])
    assert report.status == "error"
    assert report.aggregates["broken"].error_count == 5
    assert report.aggregates["broken"].mean_score is None
    assert "RuntimeError" in report.results[0].explanation


def test_pass_rate_undefined_is_not_a_pass() -> None:
    dataset = load_dataset(ROOT / "examples/datasets/synthetic.json")
    config = load_config(ROOT / "examples/suite.toml")
    config.gates = [QualityGate(evaluator="recall@3", min_pass_rate=0)]
    assert run_suite(dataset, config).status == "failed"
    config.evaluators[0].min_case_score = 0.5
    report = run_suite(dataset, config)
    assert report.status == "passed"
    assert report.aggregates["recall@3"].pass_rate == 0.5
