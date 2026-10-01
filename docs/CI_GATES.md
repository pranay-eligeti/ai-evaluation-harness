# CI gates

The workflow uses one Python 3.12 Ubuntu job. It installs the package and dev tools, runs Ruff lint/format checks, strict mypy, the complete pytest suite, and the deterministic sample. No API credentials or live provider calls are used. The base test stage excludes SDKs; an additional stage installs optional SDKs for intercepted HTTP contract tests.

Gate fields: evaluator instance name, min_mean_score, min_pass_rate, min_scored_count (default 1), max_skipped_count (optional), max_error_count (default 0). Thresholds lie in [0,1]; counts are nonnegative integers. A missing mean/pass rate fails its configured threshold even if that threshold is zero. Unknown gate targets and duplicate evaluator names are rejected.

Per-case min_case_score creates passed/failed statuses; absent thresholds produce scored statuses and no pass rate. Suite success depends on aggregate gates, not an implicit requirement that every case pass. Any execution error always makes suite status error. Empty or entirely unscored suites are insufficient_data unless a gate already fails; neither returns success.

The passing sample accepts known synthetic defects using explicit fixture thresholds. The failing configuration raises required Recall@3 to 0.9 and must exit exactly 1. CI checks that exit code and parses both JSON reports. Reports are uploaded even on job failure. A fixed threshold is a regression floor; baseline delta analysis and statistical confidence are outside Phase 1.

Phase 2's standard suite includes fake OpenAI/Anthropic request/response, dependency/credential, malformed output, error, retry, prompt-boundary, CLI opt-in, and report provenance tests. Socket connections are forbidden by an autouse test fixture. Real SDK tests skip without extras; the dedicated CI step installs providers and runs them with HTTP MockTransport. It needs no API secrets, uses nonsecret offline placeholders, and cannot incur API costs. Deterministic sample/pass/failure/report steps are preserved.

Live model behavior is not certified by offline transport tests. The optional manual CLI run requires --allow-live, environment credentials, and a model supporting the API's schema feature; it is never part of normal CI.
