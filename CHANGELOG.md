# Changelog

Implemented milestones; the first published portfolio release is `v0.3.0`.

## 0.3.0

- Application-neutral `capture-1` RAG artifacts and optional RAG Knowledge Assistant export integration.
- Baseline/candidate report comparison, case matching, added/removed cases, matched-population metric deltas, and conservative compatibility checks.
- Separate regression gates for mean scores and pass rates alongside candidate absolute quality gates.
- `ai-eval compare`, deterministic `comparison-1` JSON reports, offline examples, schemas, and CI regression checks.
- Release verification covers wheel installation, CLI entry points, and credential-free sample execution.

## 0.2.0

- Provider-neutral semantic answer relevance and groundedness judges.
- Optional OpenAI and Anthropic adapters, structured verdict validation, bounded retries, sanitized provider errors, and explicit live-call opt-in.
- Offline SDK transport tests and judge provenance in reports.
- Evaluation report schema 2 for live-provider configuration; schema 1 remains supported.

## 0.1.0

- Strict evaluation cases and JSON/JSONL datasets.
- Recall@K, Precision@K, reciprocal rank, citation metrics, answer matching, and lexical groundedness.
- TOML suites, per-case results, aggregates, absolute quality gates, JSON reports, and the `ai-eval` CLI.
- pytest, Ruff, strict mypy, and offline GitHub Actions validation.

## Artifact compatibility

Evaluation reports retain schema versions `1` and `2`. Capture and comparison
artifacts use independent versions `capture-1` and `comparison-1`. Comparison
validates report integrity and evaluator/population/judge compatibility; a schema
version alone does not establish that two runs are comparable. Future incompatible
artifact changes require an explicit schema version change.
