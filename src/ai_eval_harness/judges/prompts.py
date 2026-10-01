"""Prompt construction for LLM judges.

Prompt building is kept separate from provider invocation so that prompts can be
unit-tested -- asserted on exactly, character for character -- without any model.
Each builder returns ``None`` when the case lacks the inputs its criterion needs,
which the evaluator turns into ``SKIPPED``. A judge is never asked to assess
material that is not there.

Prompts are **versioned**. :data:`PROMPT_VERSION` is recorded in every result's
metadata, because judge scores are only comparable across runs that used the
same prompt. Changing the wording below without bumping the version would make
an evaluator-side change look like a change in the system under test.
"""

from __future__ import annotations

from typing import Protocol

from ai_eval_harness.judges.base import JudgeRequest
from ai_eval_harness.models import EvaluationCase

__all__ = [
    "ANSWER_RELEVANCE_CRITERION",
    "GROUNDEDNESS_CRITERION",
    "PROMPT_VERSION",
    "JudgePromptBuilder",
    "build_answer_relevance_request",
    "build_groundedness_request",
]

PROMPT_VERSION = "1"
"""Bump whenever any prompt text below changes. Recorded in result metadata."""

GROUNDEDNESS_CRITERION = "groundedness"
ANSWER_RELEVANCE_CRITERION = "answer_relevance"

_OUTPUT_CONTRACT = (
    "Respond with a single JSON object and nothing else. It must contain exactly "
    'these keys: "score", a number between 0.0 and 1.0, and "reasoning", a brief '
    "explanation of your score. Do not include any text outside the JSON object."
)

_GROUNDEDNESS_SYSTEM = (
    "You are a strict evaluator assessing whether an answer is supported by the "
    "provided source documents.\n"
    "Score 1.0 if every factual claim in the answer is directly supported by the "
    "sources. Score 0.0 if the answer's central claim is contradicted by, or "
    "absent from, the sources. Score in between when the answer is partially "
    "supported.\n"
    "Judge only whether the sources support the answer. Do not reward an answer "
    "for being plausible, well written, or consistent with your own knowledge.\n"
    f"{_OUTPUT_CONTRACT}"
)

_ANSWER_RELEVANCE_SYSTEM = (
    "You are a strict evaluator assessing whether an answer actually addresses "
    "the question that was asked.\n"
    "Score 1.0 if the answer directly and completely addresses the question. "
    "Score 0.0 if it addresses a different question, or declines without "
    "answering. Score in between when it is partially responsive.\n"
    "Judge only relevance to the question. Do not judge whether the answer is "
    "factually correct; that is assessed separately.\n"
    f"{_OUTPUT_CONTRACT}"
)


class JudgePromptBuilder(Protocol):
    """Builds a judging request from a case, or declines.

    Implementations must be pure and deterministic: the same case must always
    produce the same request.
    """

    def __call__(self, case: EvaluationCase) -> JudgeRequest | None:
        """Build a request for ``case``, or return ``None`` if it cannot be judged."""
        ...


def _format_documents(case: EvaluationCase) -> str:
    """Render retrieved documents that have text, in rank order, as a stable block."""
    texts = case.document_text_by_id()
    lines = [
        f"[{doc_id}] {texts[doc_id]}"
        for doc_id in dict.fromkeys(case.retrieved_document_ids)
        if doc_id in texts
    ]
    return "\n".join(lines)


def build_groundedness_request(case: EvaluationCase) -> JudgeRequest | None:
    """Build a groundedness (faithfulness) judging request.

    Args:
        case: The case to judge.

    Returns:
        A request, or ``None`` when the case has no generated answer or no
        retrieved document text -- in either situation there is nothing whose
        grounding could be assessed.
    """
    if case.generated_answer is None:
        return None
    documents = _format_documents(case)
    if not documents:
        return None

    user_prompt = (
        f"Question:\n{case.question}\n\n"
        f"Source documents:\n{documents}\n\n"
        f"Answer to evaluate:\n{case.generated_answer}"
    )
    return JudgeRequest(
        criterion=GROUNDEDNESS_CRITERION,
        case_id=case.case_id,
        system_prompt=_GROUNDEDNESS_SYSTEM,
        user_prompt=user_prompt,
        metadata={"prompt_version": PROMPT_VERSION},
    )


def build_answer_relevance_request(case: EvaluationCase) -> JudgeRequest | None:
    """Build an answer-relevance judging request.

    Args:
        case: The case to judge.

    Returns:
        A request, or ``None`` when the case has no generated answer. No
        reference answer or retrieved context is needed: relevance is judged
        against the question alone.
    """
    if case.generated_answer is None:
        return None

    user_prompt = f"Question:\n{case.question}\n\nAnswer to evaluate:\n{case.generated_answer}"
    return JudgeRequest(
        criterion=ANSWER_RELEVANCE_CRITERION,
        case_id=case.case_id,
        system_prompt=_ANSWER_RELEVANCE_SYSTEM,
        user_prompt=user_prompt,
        metadata={"prompt_version": PROMPT_VERSION},
    )
