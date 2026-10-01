import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"}
    }
    return subprocess.run(
        [sys.executable, "-m", "ai_eval_harness", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_help_and_inventory() -> None:
    assert "--dataset" in invoke("run", "--help").stdout
    assert invoke("--help").returncode == 0
    assert "llm_judge_groundedness" in invoke("list-evaluators").stdout


@pytest.mark.parametrize(
    ("config", "exit_code", "status"),
    [
        ("suite.toml", 0, "passed"),
        ("failing.toml", 1, "failed"),
    ],
)
def test_cli_gates_and_json(tmp_path: Path, config: str, exit_code: int, status: str) -> None:
    output = tmp_path / "report.json"
    result = invoke(
        "run",
        "--dataset",
        "examples/datasets/synthetic.json",
        "--config",
        f"examples/{config}",
        "--report",
        str(output),
    )
    assert result.returncode == exit_code, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == status
    assert len(report["results"]) == 40
    assert report["dataset"]["checksum"]


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--dataset", "missing.json"],
        ["run", "--dataset", "examples/datasets/synthetic.json", "--evaluator", "unknown"],
        ["run", "--dataset", "examples/datasets/synthetic.json", "--k", "0"],
        ["run", "--dataset", "examples/datasets/synthetic.json", "--config", "missing.toml"],
    ],
)
def test_cli_input_errors(args: list[str]) -> None:
    assert invoke(*args).returncode == 2


def test_cli_selection() -> None:
    result = invoke(
        "run", "--dataset", "examples/datasets/synthetic.json", "--evaluator", "citation_f1"
    )
    assert result.returncode == 0
    assert "citation_f1" in result.stdout and "recall@" not in result.stdout


def test_cli_judge_failure_exit(tmp_path: Path) -> None:
    fixture = tmp_path / "judge.json"
    fixture.write_text('{"responses": {}}')
    config = tmp_path / "suite.toml"
    config.write_text(
        'judge_responses = "judge.json"\n[[evaluators]]\ntype = "llm_judge_answer_relevance"\n'
    )
    output = tmp_path / "report.json"
    result = invoke(
        "run",
        "--dataset",
        "examples/datasets/synthetic.json",
        "--config",
        str(config),
        "--report",
        str(output),
    )
    assert result.returncode == 3
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["aggregates"]["llm_judge_answer_relevance"]["error_count"] == 5
