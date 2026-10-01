# AI Evaluation Harness

Reusable Python infrastructure for evaluating captured RAG and LLM outputs. Phase 1 provides deterministic metrics, offline judge plumbing, repeatable suites, structured reports, and CI quality gates. It does not generate answers or run a retriever.

## Installation and use

Requires Python 3.12+.

```sh
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install -e ".[dev]"
ai-eval --help
ai-eval list-evaluators
ai-eval run --dataset examples/datasets/synthetic.json --config examples/suite.toml --report scratch/report.json
ai-eval run --dataset examples/datasets/synthetic.json --config examples/failing.toml --report scratch/failing.json
```

The first suite passes; the deliberately strict second suite exits 1. Exit codes: 0 passed, 1 failed gate or insufficient measurements, 2 invalid input/configuration or report I/O, 3 evaluator execution errors. `--evaluator` may be repeated without a config; `--k` sets the cutoff for that mode.

## Architecture and supported evaluations

`models` validates captured inputs; `datasets` loads JSON/JSONL atomically; pure `metrics` feed `evaluators`; `judges` separates requests, provider calls, and verdict parsing; `runner` aggregates each evaluator and applies gates; `reporting` emits stable JSON; `cli` coordinates file I/O.

Supported deterministic evaluators: Recall@K, Precision@K (two explicit denominator conventions), reciprocal rank (suite mean is MRR), citation F1, citation validity, normalized answer exact match, token F1, and explicitly labeled lexical groundedness. Semantic answer relevance and groundedness use provider-neutral LLM judge interfaces. The included scripted provider replays fixtures; it proves execution infrastructure, not model quality. No production provider, credentials, or network calls are required.

Reports contain schema/harness versions, dataset checksum and case metadata, resolved suite configuration, evaluator provenance, every case result, separate aggregates, gate explanations, and suite status. No single combined quality score is produced. Output is deterministic for identical dataset paths, bytes, configuration, and offline providers. JSON schemas can be obtained through `SuiteReport.model_json_schema()` and `EvaluationCase.model_json_schema()`.

## Validation

```sh
python -m pytest
ruff check src tests
ruff format --check src tests
mypy
```

Unit, integration, and subprocess CLI tests exercise the real synthetic dataset. GitHub Actions installs the package on Python 3.12, runs these checks, and verifies passing and failing sample gates.

## Quality gates and extensions

TOML `[[evaluators]]` entries specify `type`, `params`, and optional `min_case_score`. `[[gates]]` reference the evaluator instance name and configure minimum mean/pass rate/scored count and maximum skipped/error count. Scores and thresholds must be finite within [0,1]. Per-case verdicts are distinct from aggregate suite gates. Any execution error makes the suite an error even without gates.

Implement the `Evaluator` protocol or subclass `BaseEvaluator`; programmatic callers can pass instances to `run_suite`. Register new types for configuration/CLI use. Implement `JudgeProvider.complete` for future provider adapters. `judge_responses` in TOML points to a scripted JSON fixture relative to the config file.

## Limitations

Lexical overlap cannot measure semantic truth, negation, paraphrases, or citation claim support. Source ID correctness depends on human annotations. Model judgments carry uncertainty and require calibration. Frozen input models prevent attribute reassignment but do not deeply freeze nested lists/dictionaries; callers must not mutate inputs during runs. Phase 1 is synchronous and in-memory, with no dashboard, hosted API, database, or live provider integration. Thresholds in the synthetic sample are regression fixtures, not production quality recommendations.

See [architecture](docs/ARCHITECTURE.md), [methodology](docs/EVALUATION_METHODOLOGY.md), [metric formulas](docs/METRICS.md), [judges](docs/LLM_JUDGES.md), and [CI gates](docs/CI_GATES.md).
