# LLM judges

Model-based evaluation contains evaluator uncertainty; it is not deterministic ground truth. Models can be biased, inconsistent, sensitive to prompt changes, and fooled by instructions embedded in evaluated content. Human calibration and adversarial evaluation are required before using judgments for high-stakes gates.

`JudgeRequest` holds criterion, case ID, system/user prompts, and metadata. `JudgeProvider` supplies name/model and `complete(request)` returning raw `JudgeResponse`. Prompt builders independently support answer relevance and groundedness, returning None when required inputs are absent. `parse_verdict` requires JSON score in [0,1] and nonblank reasoning, accepting a single surrounding Markdown fence. Boolean verdicts map to 1/0; extra payload keys are preserved in the parsed verdict. Malformed responses become visible errors, not guessed scores.

`LlmJudgeEvaluator` normalizes the verdict into the common result contract and records provider, model, criterion, prompt version, and reasoning. Prompt version must change when prompt text changes. The runner can aggregate and gate these scores, but provenance always identifies them as llm_judge.

Only offline `ScriptedJudgeProvider` and refusing `NullJudgeProvider` ship in Phase 1. Scripted models use a scripted: prefix; fixtures are test doubles, not semantic measurements. Set `judge_responses` to a JSON file containing `model` and `responses` keyed by criterion then case ID. Missing fixtures fail visibly. Live OpenAI/Anthropic/compatible adapters are future work and must implement the narrow protocol, including transport failures and model provenance, without embedding provider logic in evaluators.
