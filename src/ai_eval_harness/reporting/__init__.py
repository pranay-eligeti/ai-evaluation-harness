"""Stable JSON output; no timestamps or random run identifiers."""

import json
from pathlib import Path

from ai_eval_harness.runner.suite import SuiteReport


def report_json(report: SuiteReport) -> str:
    return (
        json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_report(report: SuiteReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json(report), encoding="utf-8")


def summary(report: SuiteReport) -> str:
    lines = [f"Suite {report.dataset['name']}: {report.status}"]
    for name, values in report.aggregates.items():
        score = "undefined" if values.mean_score is None else f"{values.mean_score:.4f}"
        lines.append(
            f"  {name}: mean={score}, scored={values.scored_count}, "
            f"skipped={values.skipped_count}, errors={values.error_count}"
        )
    lines.extend(
        f"  Gate {gate.evaluator}: {'PASS' if gate.passed else 'FAIL'} ({gate.explanation})"
        for gate in report.gates
    )
    return "\n".join(lines)
