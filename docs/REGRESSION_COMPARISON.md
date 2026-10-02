# Captured RAG runs and regression comparison

The RAG application exports outputs. The harness evaluates those outputs, then
compares evaluation reports. Neither package imports the other. No retriever,
generation service, model download, database, or network is needed in the harness.

## Capture contract

`CapturedRun` uses schema `capture-1`, independently of evaluation-report schemas
`1` and `2`. It contains `run_id`, optional `system`, `revision`, `created_at`, run
`metadata`, and a nonempty `cases` collection. The case fields are the existing
strict `EvaluationCase` contract: `case_id`, `question`, optional reference and
generated answers, ordered retrieved documents, expected document/citation IDs,
actual citation IDs, and metadata. Unknown fields, duplicate case IDs, blank IDs,
malformed types and nonfinite values are rejected. Missing annotation (`null`)
differs from an annotated empty list. JSON is the versioned run container; the
existing case-only JSONL dataset interface remains supported unchanged.

Retrieved list position is rank; explicit `rank` fields are invalid. `doc_id`
identifies the evaluation unit. Use source IDs for source-level retrieval or
chunk IDs for chunk-level retrieval; annotations and citations must use the same
unit. Document metadata may carry `source_id` and `chunk_id` (nonblank strings),
offsets and other provenance. The harness never re-sorts captured results.

```bash
ai-eval run --capture examples/rag/baseline.json --config examples/rag/suite.toml --report scratch/baseline.json
ai-eval run --capture examples/rag/candidate.json --config examples/rag/suite.toml --report scratch/candidate.json
ai-eval compare --baseline scratch/baseline.json --candidate scratch/candidate.json --config examples/rag/regression-pass.toml --report scratch/comparison.json
```

## Compatibility and matching

Comparison is conservative: instance names match exactly, never by guessed metric
similarity. Types, normalized parameters (including defaults), evaluator kind and
per-case thresholds must agree. Recall@3 versus Recall@5, differing precision
denominators, and exact match versus token F1 are not comparable. Missing metrics
are retained with `not_comparable` and a reason; they are not silently skipped.
Reports are checked for duplicate/unknown results, a complete case/evaluator grid,
and aggregates consistent with the underlying results.

Each new report records a SHA-256 identity per case over question and annotations,
excluding captured outputs. Matching IDs with different identities cannot produce
score deltas. Historical v1/v2 reports still parse unchanged; without identity
metadata, comparison requires an identical nonempty dataset checksum. A checksum
is provenance, not authentication: only compare trusted report artifacts.

Exact population equality is the default. Added and removed IDs remain visible,
and matched case diffs are still shown, but aggregate deltas are unavailable when
exact populations differ. Set `exact_population = false` in the comparison TOML
to aggregate the matched intersection only. Both reports' full case counts,
matched count, additions/removals and the aggregation semantics are explicit.
Matched scored-case coverage must agree; skipped/error results never become zero.
Undefined mean/pass-rate values remain `null` and cannot satisfy a gate.

Semantic comparisons additionally require one consistent criterion, provider,
configured model, response model and prompt version across matched scored cases,
equal on both sides. Missing or changing judge provenance is incompatible. This
also works for scripted offline judgments; it does not imply live model scores
are reproducible or statistically significant.

## Absolute quality versus regression gates

The candidate report retains its absolute gates, for example Recall@1 >= 0.6.
Regression gates separately bound a drop from the baseline:

```toml
exact_population = true
[[regression_gates]]
evaluator = "recall@1"
metric = "mean_score" # default; alternatively "pass_rate"
max_drop = 0.02
```

`max_drop` must be finite within [0, 1]. The evaluator must exist. A gate passes
when candidate minus baseline is at least `-max_drop`; equality passes (with a
1e-12 floating-point boundary allowance). Zero tolerance prohibits a decrease.
An unavailable or incompatible delta never passes. Improving can still fail an
absolute gate; meeting an absolute threshold can still fail a regression gate.
Baseline absolute failures do not disqualify an otherwise valid comparison.

## Reports and exit codes

`comparison-1` is a separate model and schema; evaluation v1/v2 meanings are
preserved. It records configuration, dataset/capture identification, compatibility,
population, separate mean and pass-rate deltas, matched/added/removed case records,
original per-case results (scores, verdicts, explanations, provenance), regression
gates, candidate absolute gates, final status, and matching/aggregation provenance.
Every numeric comparison includes baseline, candidate, delta and one of improved,
unchanged, regressed, not_comparable. No overall synthetic quality score is used.
Case/evaluator ordering and JSON serialization are stable. No clock is consulted.

CLI exit codes: 0 passed; 1 quality/regression failure; 2 malformed input/config;
3 comparison execution error, no overlap, or structured incompatibility. Incompatible
metrics produce a JSON report with reasons and `error` status. A no-overlap execution
error has no comparison artifact. Existing `run` behavior remains compatible.

## RAG Knowledge Assistant integration and fixtures

In the separate [RAG repository](https://github.com/pranay-eligeti/rag-knowledge-assistant),
run its optional exporter:

```bash
python -m src.capture --run-id rag-tfidf-baseline --top-k 1 --revision YOUR_COMMIT --output capture.json
```

The exporter uses the existing TF-IDF retriever and generator, forces offline
extractive generation, and emits source-level metric IDs with chunk provenance.
Its defaults use the repository's public synthetic corpus and annotated questions.
The API still uses TF-IDF; the separate dense adapter is unchanged. Copy the JSON
artifact to the harness; no cross-repository Python dependency is installed.

`examples/rag/baseline.json` was produced by that exporter. The candidate fixture
is explicitly a controlled perturbation of captured outputs, not a new retrieval
implementation: one retrieval improves, another regresses, and one stays unchanged.
Their aggregate recall changes cancel while citation F1 drops. The tolerant example
passes; `regression-fail.toml` rejects the citation drop with exit 1. Never interpret
these three synthetic cases as a benchmark of general RAG performance. Fixture
provenance and regeneration are described in `examples/rag/README.md`.

## Limits

This is paired deterministic artifact comparison, not statistical inference or
experiment tracking. Exact evaluator-name matching is intentional. Changes to
annotations, score coverage or judge configuration require a new valid baseline.
Partial mode does not measure added/removed cases in deltas, although candidate
absolute gates retain their original full-population semantics. Aggregate gates
can hide offsetting case regressions; inspect case diffs. Captures may contain
sensitive text: export only public/synthetic fixtures into this repository.
