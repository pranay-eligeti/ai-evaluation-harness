# LLM judges

Model-judged evaluation contains evaluator uncertainty and is not objective ground truth. Calibrate against human reviews, measure repeated-run agreement, and compare identical model and prompt versions. Models can be biased and sensitive to ordering, phrasing, and prompt injection. Structured output establishes syntax, not correctness.

## Architecture and providers

`JudgeRequest` carries criterion, case ID, system/user prompts, and metadata. `JudgeProvider.complete` returns raw `JudgeResponse`; transport lives outside evaluator logic. `LlmJudgeEvaluator` parses a verdict, creates the shared result, and retains provenance even on errors/skips. `ScriptedJudgeProvider` and `NullJudgeProvider` remain offline options.

The optional OpenAI adapter uses `client.responses.create`, instructions/input, native JSON schema format, explicit model/timeout/output-token limit, and `store=False`. It rejects incomplete/failed responses and missing output text/model. The Anthropic adapter uses `client.messages.create`, system/user messages, `output_config.format` JSON schema, and explicit max_tokens. It accepts only one text block with end_turn, rejecting refusals, truncation, and unexpected blocks. No fallback to free-form output or another model occurs.

Supported SDK ranges are OpenAI >=3.22.1,<4 and Anthropic >=1.11.0,<2. Version 3.22.1 and 1.11.0 were verified locally using intercepted HTTP, including success, authentication errors, and timeouts. No live/paid call was made; account model access and server-side schema compatibility must be checked by the user. Choose a model supporting native structured output. Different providers may use different available model names.

Official API patterns: [OpenAI SDK](https://github.com/openai/openai-python) and [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). Provider SDKs are lazy-loaded only when constructed. Base imports/tests require neither SDK. Install the openai, anthropic, or providers extra. Missing SDK errors name the extra; missing credential errors name OPENAI_API_KEY or ANTHROPIC_API_KEY.

## Configuration and explicit live opt-in

```toml
[judge]
provider = "openai" # or "anthropic"
model = "YOUR_STRUCTURED_OUTPUT_MODEL"
timeout = 30.0
max_retries = 1
retry_delay = 1.0
max_output_tokens = 1024
prompt_version = "2"

[[evaluators]]
type = "llm_judge_answer_relevance"
[[evaluators]]
type = "llm_judge_groundedness"
```

Use `--allow-live` with a live suite. Calls cost money and disclose evaluated content to the provider; ensure you are authorized to send it. CLI credentials come exclusively from runtime environment. Programmatic `build_provider` accepts api_key or an injected client, never storing keys in suite configuration. SDK environment endpoint overrides are not honored; constructed clients use official endpoints. Custom endpoints are outside Phase 2.

Timeout is a positive finite SDK per-request timeout up to 300 seconds, not a total suite deadline. Retry count is 0..3; fixed retry_delay is 0..10 seconds. Output limit is 128..8192 tokens. Provider/model/prompt-version values are validated. Only prompt version 2 is supported by current live config; older prompt text is not silently selected. Choose either live judge or judge_responses, not both.

## Version 2 rubrics and data boundary

User messages are deterministic JSON objects containing question and answer, plus nonblank retrieved document text for faithfulness. JSON escaping prevents content from breaking out of its field syntactically; the system prompt identifies all fields as untrusted data and tells the model not to follow embedded requests or reveal instructions. This does not guarantee semantic injection resistance. Prompt version is bumped whenever rubric text changes.

Answer relevance asks whether the response meets the question's information need, independently of truth or grounding. Anchors: 1 complete/direct, .75 minor omissions, .5 partially addressed, .25 mostly tangential, 0 unrelated/empty/generic refusal. A justified explanation of inability to answer can be relevant. A wrong factual answer can still address the question.

Groundedness uses supplied context only. Anchors: 1 all factual claims supported, .75 main claims supported/minor unsupported detail, .5 mixed support, .25 little supported, 0 central claim absent/contradicted. Contradictions weigh heavily; external knowledge must not fill gaps. The rubric asks for supported/unsupported/contradicted distinctions and insufficient-evidence explanations. No factual claims score 0; evidence-consistent uncertainty can be supported. Missing generated answer or missing/blank context skips; relevant context that fails to support a claim is assessed rather than skipped.

## Response and error contracts

Live/evaluator verdicts require exactly numeric score in [0,1] and nonblank string reasoning. Boolean scores, extra/duplicate keys, nonfinite scores, malformed JSON, fences, missing fields, and wrong types are rejected. There is no score recovery. The standalone legacy parse_verdict default retains Phase 1 boolean/fence/extra-field behavior; strict=True is used for live adapters and semantic evaluators.

Configuration errors (including missing SDK/credential) occur before execution and produce CLI exit 2. Runtime normalized ProviderAuthenticationError and ProviderRequestError are permanent; ProviderRateLimitError, ProviderTimeoutError, ProviderConnectionError, and ProviderServerError are retryable. Only 429, 5xx, timeout, and connection failures retry. Other request errors, malformed/refused/incomplete output, and verdict validation never retry. SDK automatic retries are disabled, including on injected clients through with_options, to avoid multiplied attempts. No unbounded backoff or retry-after waits are used; the maximum is 1+max_retries attempts. A timeout retry may incur a second charge if the first request completed server-side.

After exhaustion, failures remain case-level ERROR with no numeric score, and suite status error/CLI exit 3. SDK exception strings, raw headers, raw response bodies, and credentials are not serialized. Malformed raw response remains accessible only on the exception's programmatic raw_response attribute, not str(exception) or persisted error metadata. This attribute can contain sensitive output; do not log it.

## Reports and limitations

Metadata records criterion, judge_provider, actual judge_model on success, configured_model, prompt_version, judge_reasoning, and attempts. Errors/skips retain configured model/provider/criterion/prompt version. Config-driven live reports use schema version 2; old deterministic/scripted version 1 reports remain readable and keep their configuration shape.

Normal tests forbid socket connections. Adapter fakes run without SDKs; optional real SDK tests use HTTP MockTransport, never live requests. CI installs extras only to exercise these offline tests. Live verification is the explicitly opted-in CLI path; there are no automatic live tests.

Reasoning and case metadata can reproduce sensitive evaluated content. Minimize datasets, restrict report access, and disable SDK/HTTP debug logging. OpenAI store=False is not a claim of zero provider retention; provider policies govern data processing. Models, availability, and server behavior change over time. Synchronous in-memory execution, no response cache, no token-cost accounting, and no statistical calibration engine are deliberate Phase 2 limits.
