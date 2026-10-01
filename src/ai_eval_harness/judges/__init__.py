"""Provider-neutral LLM-as-a-judge abstraction.

Model-based evaluation is split into four independent pieces so that none of
them is coupled to a particular vendor, and each can be tested on its own:

1. **Prompt construction** (:mod:`ai_eval_harness.judges.prompts`) turns an
   evaluation case into a :class:`JudgeRequest`. Pure; no provider involved.
2. **The provider interface** (:mod:`ai_eval_harness.judges.base`) is a
   :class:`JudgeProvider` protocol with a single ``complete`` method. An adapter
   for OpenAI, Anthropic, or any other service is an implementation of this
   protocol and lives outside the evaluation logic entirely.
3. **Response parsing** (:mod:`ai_eval_harness.judges.parsing`) turns raw provider
   text into a :class:`JudgeVerdict`, or raises. Pure.
4. **Result normalisation** happens in
   :class:`ai_eval_harness.evaluators.llm_judge.LlmJudgeEvaluator`, which maps a
   verdict onto the same :class:`~ai_eval_harness.results.EvaluationResult` type
   every deterministic evaluator produces.

**The default path remains offline.** Two offline providers are included:
:class:`ScriptedJudgeProvider`, which replays recorded verdicts and makes the
judge path genuinely executable offline and in CI, and
:class:`NullJudgeProvider`, which refuses. Neither invents a score. Optional OpenAI
and Anthropic adapters live in ``judges.providers`` and are constructed explicitly.
Importing this package does not load their SDKs. See ``docs/LLM_JUDGES.md``.
"""

from ai_eval_harness.judges.base import (
    JudgeProvider,
    JudgeRequest,
    JudgeResponse,
    JudgeVerdict,
)
from ai_eval_harness.judges.parsing import parse_verdict
from ai_eval_harness.judges.prompts import (
    ANSWER_RELEVANCE_CRITERION,
    GROUNDEDNESS_CRITERION,
    JudgePromptBuilder,
    build_answer_relevance_request,
    build_groundedness_request,
)
from ai_eval_harness.judges.scripted import (
    NullJudgeProvider,
    ScriptedJudgeProvider,
    load_scripted_provider,
)

__all__ = [
    "ANSWER_RELEVANCE_CRITERION",
    "GROUNDEDNESS_CRITERION",
    "JudgePromptBuilder",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeResponse",
    "JudgeVerdict",
    "NullJudgeProvider",
    "ScriptedJudgeProvider",
    "build_answer_relevance_request",
    "build_groundedness_request",
    "load_scripted_provider",
    "parse_verdict",
]
