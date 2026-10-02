"""Pure, conservative comparison of evaluation reports, never of RAG internals."""

from pathlib import Path
from typing import Any, Literal

from ai_eval_harness.comparison_models import (
    CaseChange,
    ComparisonConfig,
    ComparisonReport,
    Delta,
    MetricComparison,
    Population,
    RegressionResult,
)
from ai_eval_harness.evaluators.registry import build_evaluator, get_specification
from ai_eval_harness.results import EvaluationResult, EvaluatorKind
from ai_eval_harness.runner.suite import SuiteReport, aggregate, apply_quality_gates


class ComparisonError(Exception):
    """Valid inputs cannot be compared or execution failed (CLI exit 3)."""


def delta(left: float | None, right: float | None, reason: str = "") -> Delta:
    if reason or left is None or right is None:
        return Delta(baseline=left, candidate=right, reason=reason or "Undefined score")
    change = right - left
    return Delta(
        baseline=left,
        candidate=right,
        delta=change,
        status="improved" if change > 0 else "regressed" if change < 0 else "unchanged",
    )


def load_report(path: Path) -> SuiteReport:
    report = SuiteReport.model_validate_json(path.read_text(encoding="utf-8"))
    validate_report(report)
    return report


def validate_report(report: SuiteReport) -> None:
    """Check internal integrity not enforced by the historical report parser."""
    if any(set(item) != {"name", "type", "kind"} for item in report.evaluators):
        raise ValueError("Report evaluator descriptors require name, type and kind")
    names = [item["name"] for item in report.evaluators]
    population = report.dataset.get("case_metadata")
    if not isinstance(population, dict) or any(
        not isinstance(key, str) or not key.strip() for key in population
    ):
        raise ValueError("Report case metadata must be an object with nonblank case IDs")
    identities = report.dataset.get("case_identity", {})
    if not isinstance(identities, dict):
        raise ValueError("Report case identity must be an object")
    ids = list(population)
    if not names or len(names) != len(set(names)):
        raise ValueError("Report has missing or duplicate evaluator identities")
    if report.dataset.get("case_count") != len(ids) or not ids:
        raise ValueError("Report population metadata is missing or inconsistent")
    if len(report.configuration.evaluators) != len(names):
        raise ValueError("Report evaluator configuration is inconsistent")
    signatures(report)
    seen: set[tuple[str, str]] = set()
    for result in report.results:
        key = result.case_id, result.evaluator
        if key in seen or result.case_id not in ids or result.evaluator not in names:
            raise ValueError("Report has duplicate or unknown case/evaluator results")
        descriptor = report.evaluators[names.index(result.evaluator)]
        evaluator_config = report.configuration.evaluators[names.index(result.evaluator)]
        if result.threshold != evaluator_config.min_case_score:
            raise ValueError("Result threshold disagrees with evaluator configuration")
        if (result.evaluator_type, result.kind.value) != (descriptor["type"], descriptor["kind"]):
            raise ValueError("Result identity disagrees with evaluator descriptor")
        seen.add(key)
    if len(seen) != len(ids) * len(names):
        raise ValueError("Report is missing case/evaluator results")
    if set(report.aggregates) != set(names):
        raise ValueError("Report is missing expected aggregate")
    for name in names:
        computed = aggregate([r for r in report.results if r.evaluator == name])
        if computed != report.aggregates[name]:
            raise ValueError("Report aggregate disagrees with case results")

    if any(g.evaluator not in names for g in report.configuration.gates):
        raise ValueError("Quality gate references an absent evaluator")
    if report.gates != apply_quality_gates(report.aggregates, report.configuration.gates):
        raise ValueError("Absolute gate results disagree with configuration")
    expected_status = (
        "error"
        if any(r.status.value == "error" for r in report.results)
        else "failed"
        if any(not g.passed for g in report.gates)
        else "insufficient_data"
        if not any(r.score is not None for r in report.results)
        else "passed"
    )
    if report.status != expected_status:
        raise ValueError("Report status disagrees with results and gates")


def signatures(report: SuiteReport) -> dict[str, Any]:
    signatures: dict[str, Any] = {}
    for descriptor, config in zip(report.evaluators, report.configuration.evaluators, strict=True):
        if descriptor["type"] != config.type:
            raise ValueError("Report configuration type disagrees with evaluator")
        spec = get_specification(config.type)
        if descriptor["kind"] != spec.kind.value:
            raise ValueError("Report evaluator kind disagrees with registry")
        expected_name = (
            config.type
            if spec.requires_judge
            else build_evaluator(config.type, config.params, config.min_case_score).name
        )
        if descriptor["name"] != expected_name:
            raise ValueError("Report evaluator name disagrees with configuration")
        if set(config.params) - {parameter.name for parameter in spec.parameters}:
            raise ValueError("Unknown evaluator parameter in report")
        if any(p.required and p.name not in config.params for p in spec.parameters):
            raise ValueError("Missing required evaluator parameter in report")
        params = {param.name: param.default for param in spec.parameters}
        params.update(config.params)
        signatures[descriptor["name"]] = (
            config.type,
            descriptor["kind"],
            params,
            config.min_case_score,
        )
    return signatures


def judge_signature(results: list[EvaluationResult]) -> set[tuple[str, ...]] | None:
    keys = ("criterion", "judge_provider", "configured_model", "judge_model", "prompt_version")
    values: set[tuple[str, ...]] = set()
    for result in results:
        if result.score is None:
            continue
        if any(not isinstance(result.metadata.get(key), str) for key in keys):
            return None
        values.add(tuple(str(result.metadata[key]) for key in keys))
    return values if len(values) == 1 else None


def compare_reports(
    baseline: SuiteReport, candidate: SuiteReport, config: ComparisonConfig
) -> ComparisonReport:
    validate_report(baseline)
    validate_report(candidate)
    left_ids = set(baseline.dataset["case_metadata"])
    right_ids = set(candidate.dataset["case_metadata"])
    matched = sorted(left_ids & right_ids)
    if not matched:
        raise ComparisonError("No overlapping cases")
    left_sig, right_sig = signatures(baseline), signatures(candidate)
    names = sorted(left_sig.keys() | right_sig.keys())
    if any(g.evaluator not in names for g in config.regression_gates):
        raise ValueError("Regression gate references an absent evaluator")
    left = {(r.case_id, r.evaluator): r for r in baseline.results}
    right = {(r.case_id, r.evaluator): r for r in candidate.results}
    identities_left = baseline.dataset.get("case_identity", {})
    identities_right = candidate.dataset.get("case_identity", {})
    same_bytes = (
        not identities_left
        and not identities_right
        and bool(baseline.dataset.get("checksum"))
        and (baseline.dataset.get("checksum") == candidate.dataset.get("checksum"))
    )
    identity_ok = {
        case_id: (
            bool(identities_left.get(case_id))
            and (identities_left.get(case_id) == identities_right.get(case_id))
        )
        or same_bytes
        for case_id in matched
    }
    population_ok = not config.exact_population or left_ids == right_ids
    metrics: dict[str, MetricComparison] = {}
    case_metrics: dict[str, dict[str, Delta]] = {i: {} for i in left_ids | right_ids}
    for name in names:
        reason = ""
        if name not in left_sig or name not in right_sig:
            reason = "Evaluator missing on one side"
        elif left_sig[name] != right_sig[name]:
            reason = "Evaluator configuration mismatch"
        lr = [left[(i, name)] for i in matched if (i, name) in left]
        rr = [right[(i, name)] for i in matched if (i, name) in right]
        if (
            not reason
            and lr[0].kind == EvaluatorKind.LLM_JUDGE
            and (judge_signature(lr) is None or judge_signature(lr) != judge_signature(rr))
        ):
            reason = "Judge criterion/provider/model/prompt provenance mismatch or missing"
        for case_id in sorted(case_metrics):
            a, b = left.get((case_id, name)), right.get((case_id, name))
            case_reason = reason
            if case_id not in matched:
                case_reason = "Added or removed case"
            elif not identity_ok[case_id]:
                case_reason = "Question/annotation identity mismatch or unavailable"
            case_metrics[case_id][name] = delta(
                a.score if a else None, b.score if b else None, case_reason
            )
        if not reason and not all(identity_ok.values()):
            reason = "Question/annotation identity mismatch or unavailable"
        if not reason and not population_ok:
            reason = "Exact population required; added or removed cases"
        if not reason and (
            {r.case_id for r in lr if r.score is not None}
            != {r.case_id for r in rr if r.score is not None}
        ):
            reason = "Scored coverage differs on matched cases"
        la, ra = aggregate(lr), aggregate(rr)
        metrics[name] = MetricComparison(
            compatible=not reason,
            reason=reason,
            mean_score=delta(la.mean_score, ra.mean_score, reason),
            pass_rate=delta(la.pass_rate, ra.pass_rate, reason),
        )
    if not any(m.compatible and m.mean_score.delta is not None for m in metrics.values()):
        # Return structured incompatibility when metrics exist, not fabricated regressions.
        compatible = False
    else:
        compatible = all(m.compatible for m in metrics.values())
    gates: list[RegressionResult] = []
    for gate in config.regression_gates:
        value = getattr(metrics[gate.evaluator], gate.metric)
        passed = value.delta is not None and value.delta >= -gate.max_drop - 1e-12
        gates.append(
            RegressionResult(
                evaluator=gate.evaluator,
                metric=gate.metric,
                max_drop=gate.max_drop,
                passed=passed,
                explanation=value.reason or (f"delta={value.delta}; required >= {-gate.max_drop}"),
            )
        )
    error = not compatible or baseline.status == "error" or candidate.status == "error"
    status: Literal["passed", "failed", "error"] = (
        "error"
        if error
        else "failed"
        if (any(not g.passed for g in gates) or candidate.status != "passed")
        else "passed"
    )
    return ComparisonReport(
        baseline=baseline.dataset,
        candidate=candidate.dataset,
        configuration=config,
        compatible=compatible,
        population=Population(
            baseline_count=len(left_ids),
            candidate_count=len(right_ids),
            matched_count=len(matched),
            added=sorted(right_ids - left_ids),
            removed=sorted(left_ids - right_ids),
            semantics="exact" if config.exact_population else "matched_only",
        ),
        evaluators=metrics,
        cases=[
            CaseChange(
                case_id=i,
                population_status="matched"
                if i in matched
                else ("added" if i in right_ids else "removed"),
                metrics=case_metrics[i],
                baseline_results={
                    n: left[(i, n)].model_dump(mode="json") for n in names if (i, n) in left
                },
                candidate_results={
                    n: right[(i, n)].model_dump(mode="json") for n in names if (i, n) in right
                },
            )
            for i in sorted(case_metrics)
        ],
        regression_gates=gates,
        candidate_absolute_gates=candidate.gates,
        status=status,
        provenance={
            "matching": "case_id plus question/annotation SHA-256",
            "aggregation": "matched population; identical scored coverage required",
        },
    )
