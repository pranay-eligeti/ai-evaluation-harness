"""The evaluation data model.

A single :class:`EvaluationCase` carries everything the harness needs to evaluate
one interaction with a RAG or LLM system: the question, the system's answer, the
context it retrieved, and whatever ground truth is available.

Three design decisions in this module are load-bearing and are relied upon
throughout the harness:

**1. Retrieval order is the ranking.**
``EvaluationCase.retrieved_documents`` is an ordered list and the harness never
re-sorts it. Rank 1 is ``retrieved_documents[0]``. If the harness sorted by
``score`` internally, every tie would be broken by an implicit, undocumented rule.
Dataset authors who have scores rather than an order should call
:func:`rank_documents`, which sorts explicitly and documents its tie-breaking.

**2. ``None`` means "not annotated"; ``[]`` means "annotated as empty".**
For every ground-truth field (``reference_answer``, ``expected_document_ids``,
``expected_citations``) the two are different facts and produce different
outcomes. ``expected_document_ids=None`` means nobody labelled this case for
retrieval, so retrieval evaluators report ``SKIPPED``.
``expected_document_ids=[]`` means a human asserted that no document in the
corpus is relevant -- a real, scoreable annotation (see ``docs/METRICS.md``).

**3. Malformed input is rejected, not repaired.**
All models set ``extra="forbid"``, so a misspelled field is an error rather than
a silently ignored key that leaves a ground-truth field empty.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "RetrievedDocument",
    "rank_documents",
]

# A non-empty, non-whitespace identifier. Identifiers are compared exactly
# (case-sensitively) everywhere in the harness; see docs/METRICS.md.
DocumentId = Annotated[str, Field(min_length=1)]

_STRICT_MODEL = ConfigDict(
    extra="forbid", frozen=True, strict=True, allow_inf_nan=False, str_strip_whitespace=False
)


def _reject_blank_ids(values: Sequence[str], field_name: str) -> list[str]:
    """Reject empty/whitespace-only identifiers rather than silently dropping them."""
    cleaned: list[str] = []
    for index, value in enumerate(values):
        if not value.strip():
            raise ValueError(f"{field_name}[{index}] is blank; identifiers must be non-empty")
        cleaned.append(value)
    return cleaned


class RetrievedDocument(BaseModel):
    """One document or chunk returned by a retrieval system.

    Attributes:
        doc_id: Stable identifier for the document or chunk. This is the value
            matched against ``expected_document_ids`` and against citations.
        text: The document's content. Optional, because retrieval-only metrics
            (Recall@K, Precision@K, MRR) need identifiers, not text. Evaluators
            that genuinely need text -- groundedness, for instance -- report
            ``SKIPPED`` when it is absent instead of scoring an empty string.
        score: The retriever's own relevance score, if any. The harness never
            uses this for ranking (see module docstring); it is carried through
            to reports so failures can be diagnosed.
        metadata: Free-form provenance (source URI, chunk offsets, ...). Never
            interpreted by the harness.
    """

    model_config = _STRICT_MODEL

    doc_id: DocumentId
    text: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("doc_id must not be blank")
        return value


def rank_documents(documents: Iterable[RetrievedDocument]) -> list[RetrievedDocument]:
    """Order documents by descending ``score``, with explicit tie-breaking.

    Provided for dataset authors whose retriever emits scores rather than an
    ordered list. The harness itself never calls this during evaluation -- the
    order stored on the case *is* the ranking.

    Ranking rules, in order:

    1. Documents with a ``score`` rank above documents without one.
    2. Among scored documents, higher ``score`` ranks first.
    3. Ties (equal scores, or two unscored documents) preserve input order.
       The sort is stable, so the caller's order is the documented tie-break.

    Args:
        documents: Documents to order.

    Returns:
        A new list; the input is not modified.
    """
    materialised = list(documents)
    scored = [doc for doc in materialised if doc.score is not None]
    unscored = [doc for doc in materialised if doc.score is None]
    # `sorted` is stable, so equal scores keep their relative input order.
    scored.sort(key=lambda doc: doc.score if doc.score is not None else float("-inf"), reverse=True)
    return scored + unscored


class EvaluationCase(BaseModel):
    """A single evaluation case: one question, one system response, and its ground truth.

    Attributes:
        case_id: Unique identifier within a dataset. Appears in every result and
            report entry so a failure can be traced back to its input.
        question: The user input given to the system under test.
        reference_answer: The gold answer, when one exists. ``None`` means the
            case is not annotated for answer quality; answer evaluators report
            ``SKIPPED`` rather than scoring against nothing.
        generated_answer: The answer the system produced. ``None`` means no
            answer was captured (``SKIPPED``). An empty string means the system
            answered with nothing, which *is* scored -- and scores 0.0.
        retrieved_documents: Documents the system retrieved, in rank order.
        expected_document_ids: The ids considered relevant for this question.
            ``None`` = unannotated (retrieval evaluators skip). ``[]`` = "no
            document in the corpus is relevant", a real annotation used for the
            missing-evidence scenario.
        expected_citations: The ids the answer should have cited. ``None`` =
            unannotated. ``[]`` = "this answer should cite nothing".
        actual_citations: The ids the system actually cited. Defaults to ``[]``
            (the system cited nothing), which is a scoreable observation rather
            than a missing annotation.
        metadata: Free-form labels (scenario name, difficulty, ...). Never
            interpreted by the harness, but carried into reports.
    """

    model_config = _STRICT_MODEL

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    generated_answer: str | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    expected_document_ids: list[DocumentId] | None = None
    expected_citations: list[DocumentId] | None = None
    actual_citations: list[DocumentId] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "question")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_document_ids", "expected_citations")
    @classmethod
    def _validate_optional_id_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _reject_blank_ids(value, "identifier list")

    @field_validator("actual_citations")
    @classmethod
    def _validate_actual_citations(cls, value: list[str]) -> list[str]:
        return _reject_blank_ids(value, "actual_citations")

    @property
    def retrieved_document_ids(self) -> list[str]:
        """Retrieved document ids in rank order, duplicates preserved.

        Duplicates are preserved here because de-duplication is a *metric*
        decision, documented and applied in :mod:`ai_eval_harness.metrics.retrieval`.
        """
        return [document.doc_id for document in self.retrieved_documents]

    def document_text_by_id(self) -> dict[str, str]:
        """Map of ``doc_id`` to text, restricted to documents that have text.

        When the same ``doc_id`` appears more than once, the first occurrence
        wins -- consistent with the retrieval metrics, which keep the highest
        (earliest) rank for a duplicated id.
        """
        texts: dict[str, str] = {}
        for document in self.retrieved_documents:
            if document.text is not None and document.doc_id not in texts:
                texts[document.doc_id] = document.text
        return texts


class EvaluationDataset(BaseModel):
    """An ordered collection of evaluation cases with unique ids.

    Attributes:
        name: Human-readable dataset name, used in reports.
        cases: The cases, in file order. Evaluation preserves this order so
            reports are stable across runs.
        source_path: Where the dataset was loaded from, if from a file.
        checksum: SHA-256 of the raw source bytes, when loaded from a file.
            Recorded in reports so a result can be tied to exact input bytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    cases: list[EvaluationCase] = Field(default_factory=list)
    source_path: str | None = None
    checksum: str | None = None

    @model_validator(mode="after")
    def _validate_unique_case_ids(self) -> EvaluationDataset:
        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.case_id in seen:
                duplicates.append(case.case_id)
            seen.add(case.case_id)
        if duplicates:
            raise ValueError("duplicate case_id(s): " + ", ".join(sorted(set(duplicates))))
        return self

    def __len__(self) -> int:
        return len(self.cases)
