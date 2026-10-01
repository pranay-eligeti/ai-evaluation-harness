# Architecture

Data flows from strict Pydantic `EvaluationCase`/`RetrievedDocument` models through an evaluator protocol into `EvaluationResult`. JSON and JSONL loaders reject malformed cases without silently shrinking the suite. Ordered documents preserve ranking; scores never trigger an implicit sort. An explicit stable `rank_documents` helper is available to dataset authors.

Pure metric functions have no file or provider I/O. `BaseEvaluator` supplies threshold handling, explanations, and error capture. The registry validates type-specific parameters. The runner adds defensive exception capture for custom evaluators, checks result identity, aggregates independently by instance, and applies gates. Duplicate instance names and unknown gate targets are configuration errors.

A result carries deterministic/LLM provenance and scored/passed/failed/skipped/error status. Undefined measurements are omitted from means, with skip counts exposed. Execution errors always prevent suite success. Empty/all-unscored suites cannot pass without measurements.

Judge prompts, provider protocol, raw response, validated verdict, and normalized evaluator results are separate modules. Scripted fixtures exercise this path without network dependencies. Reports serialize models with stable key ordering, exact source-byte SHA-256, and no wall-clock fields. Report schema version is 1.

The CLI owns config-relative scripted fixture paths and human summaries. Python callers supply `EvaluatorBuildContext` directly. The package uses only Pydantic at runtime; argparse, TOML, hashing, and JSON come from the standard library.
