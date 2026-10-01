# Architecture

Data flows from strict Pydantic `EvaluationCase`/`RetrievedDocument` models through an evaluator protocol into `EvaluationResult`. JSON and JSONL loaders reject malformed cases without silently shrinking the suite. Ordered documents preserve ranking; scores never trigger an implicit sort. An explicit stable `rank_documents` helper is available to dataset authors.

Pure metric functions have no file or provider I/O. `BaseEvaluator` supplies threshold handling, explanations, and error capture. The registry validates type-specific parameters. The runner adds defensive exception capture for custom evaluators, checks result identity, aggregates independently by instance, and applies gates. Duplicate instance names and unknown gate targets are configuration errors.

A result carries deterministic/LLM provenance and scored/passed/failed/skipped/error status. Undefined measurements are omitted from means, with skip counts exposed. Execution errors always prevent suite success. Empty/all-unscored suites cannot pass without measurements.

Judge prompts, provider protocol, raw response, validated verdict, and normalized evaluator results are separate modules. Scripted fixtures exercise this path without network dependencies. Reports serialize models with stable key ordering, exact source-byte SHA-256, and no wall-clock fields. Deterministic/scripted report schema version stays 1. Config-driven live suites use version 2 for the new judge configuration; the report model reads both.

The CLI owns config-relative scripted fixture paths and human summaries. Python callers supply `EvaluatorBuildContext` directly. The package uses only Pydantic at runtime; argparse, TOML, hashing, and JSON come from the standard library.

Phase 2 adds `judges/providers`: validated nonsecret configuration, shared bounded transport/error policy, two narrow SDK adapters, and a provider factory. No optional SDK is imported at package/module load time. SDK-shaped dynamic values are confined to this optional transport boundary; evaluators retain the existing provider protocol.

Provider clients disable their own retries; the harness retries only normalized transient failures. Validation stays outside retry handling. Constructed clients use fixed official endpoints and environment/programmatic credentials. The CLI requires --allow-live and closes owned clients with a context manager. Injected clients retain caller ownership. Provider factory construction is explicit for Python callers; run_suite still receives EvaluatorBuildContext and never initiates provider/network setup itself.

The common result/report top-level structure is preserved. Semantic metadata includes criterion/provider/model/prompt provenance, including failures and skips. Strict verdict parsing and sanitized exceptions prevent raw transport errors or malformed outputs from entering reports. Version 2 prompts encode evaluated content as JSON data and have separate relevance/faithfulness rubrics.
