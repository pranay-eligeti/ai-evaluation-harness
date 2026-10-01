"""The deterministic groundedness proxy.

This is the clearest case in the harness where a deterministic metric is *not* a
substitute for the thing it approximates. Faithfulness -- "is every claim in this
answer supported by the retrieved context?" -- is a semantic judgement. What this
evaluator measures is vocabulary overlap.

It is included anyway because it is free, reproducible, and catches the blunt
failure mode: an answer whose content words are largely absent from its own
context. It is *named* ``lexical_groundedness`` rather than ``groundedness``, and
its explanation text states its limitation on every single result, so nobody
reading a report can mistake it for a faithfulness measurement.

For genuine faithfulness, use ``llm_judge_groundedness`` and read
``docs/LLM_JUDGES.md`` on the uncertainty that introduces.
"""

from __future__ import annotations

from typing import ClassVar

from ai_eval_harness.evaluators.base import BaseEvaluator
from ai_eval_harness.metrics.text import lexical_groundedness
from ai_eval_harness.models import EvaluationCase
from ai_eval_harness.results import EvaluatorKind, Measurement

__all__ = ["LexicalGroundednessEvaluator"]

_CAVEAT = "Lexical overlap only; this is not a semantic faithfulness judgement."


class LexicalGroundednessEvaluator(BaseEvaluator):
    """Share of the answer's distinct content words that occur in the retrieved context.

    Args:
        min_case_score: Optional per-case threshold. Choose conservatively: a
            faithful paraphrase can score low, so a high threshold here produces
            false failures.
    """

    _evaluator_type: ClassVar[str] = "lexical_groundedness"
    kind: ClassVar[EvaluatorKind] = EvaluatorKind.DETERMINISTIC
    description: ClassVar[str] = (
        "Share of the answer's content words found in the retrieved context. A weak "
        "lexical proxy for faithfulness, NOT a semantic judgement."
    )

    def measure(self, case: EvaluationCase) -> Measurement:
        if case.generated_answer is None:
            return Measurement(
                score=None,
                explanation="Case has no generated_answer, so there is nothing to ground.",
            )

        texts_by_id = case.document_text_by_id()
        if not texts_by_id:
            return Measurement(
                score=None,
                explanation=(
                    "No retrieved document carries text, so grounding cannot be "
                    "assessed. An answer is not ungrounded merely because the "
                    "harness was given no document text."
                ),
                metadata={"documents_with_text": 0},
            )

        # Iterate in rank order for a stable, reviewable document list.
        ordered_ids = [
            doc_id for doc_id in dict.fromkeys(case.retrieved_document_ids) if doc_id in texts_by_id
        ]
        score = lexical_groundedness(
            case.generated_answer, [texts_by_id[doc_id] for doc_id in ordered_ids]
        )
        metadata = {
            "documents_with_text": len(ordered_ids),
            "grounding_document_ids": ordered_ids,
        }
        if score is None:
            return Measurement(
                score=None,
                explanation=(
                    "The answer contains no content words after stopword removal, "
                    "so lexical groundedness has no denominator."
                ),
                metadata=metadata,
            )

        return Measurement(
            score=score,
            explanation=(
                f"Lexical groundedness = {score:.3f}: that share of the answer's "
                f"distinct content words appears in {len(ordered_ids)} retrieved "
                f"document(s). {_CAVEAT}"
            ),
            metadata=metadata,
        )
