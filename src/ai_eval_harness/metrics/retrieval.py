"""Rank-aware retrieval metrics: Recall@K, Precision@K, and reciprocal rank.

Shared definitions
------------------

**Relevance is binary and id-based.** A retrieved document is *relevant* iff its
``doc_id`` appears in the case's ``expected_document_ids``. Ids are compared
exactly (case-sensitively, no normalisation). There are no graded relevance
levels in Phase 1; adding them would change these formulas, so they are not
faked here.

**Ranking comes from list order.** ``retrieved_ids[0]`` is rank 1. These functions
never sort, so they never introduce an implicit tie-break. Ties in a retriever's
own scores must be resolved by the caller before the list reaches this module
(see :func:`ai_eval_harness.models.rank_documents`).

**Duplicates are collapsed, keeping the best rank.** A retriever that returns the
same ``doc_id`` twice has surfaced one document, not two. Duplicates are removed
keeping the *first* (highest-ranked) occurrence, and only then is the list
truncated to K. Doing it in this order matters: with ``retrieved=[A, A, B]`` and
``k=2``, collapsing first gives ``[A, B]`` -- truncating first would give ``[A]``
and silently hide B. Duplicate *expected* ids are likewise collapsed, so a
duplicated ground-truth entry cannot inflate or deflate a denominator.

**``k`` must be at least 1.** ``k=0`` describes no meaningful retrieval question
and every metric would be either undefined or trivially zero, so it raises
``ValueError`` as a configuration bug rather than returning a number. ``k``
larger than the retrieved list is fine and common: the list is simply shorter
than K, which is exactly what the metrics need to express.

See ``docs/METRICS.md`` for the formulas, worked examples, and the rationale for
each ``None`` case.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from ai_eval_harness.metrics.aggregation import mean
from ai_eval_harness.metrics.validation import validated_ids

__all__ = [
    "PrecisionDenominator",
    "dedupe_preserving_order",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "top_k",
]

PrecisionDenominator = Literal["retrieved", "k"]
"""Which denominator Precision@K divides by. See :func:`precision_at_k`."""


def dedupe_preserving_order(ids: Iterable[str]) -> list[str]:
    """Remove duplicate ids, keeping the first (highest-ranked) occurrence.

    Args:
        ids: Ids in rank order.

    Returns:
        A new list containing each distinct id once, in first-seen order.

    Example:
        ``["a", "b", "a", "c", "b"]`` becomes ``["a", "b", "c"]``.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in validated_ids(ids):
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be >= 1, got {k}. K=0 defines no retrieval question.")


def top_k(retrieved_ids: Sequence[str], k: int) -> list[str]:
    """The effective top-K: duplicates collapsed, then truncated to ``k``.

    This is the canonical preprocessing step shared by every retrieval metric,
    so all of them agree on what "the top K results" means.

    Args:
        retrieved_ids: Retrieved document ids in rank order.
        k: Cut-off rank; must be >= 1. May exceed ``len(retrieved_ids)``, in
            which case the whole (de-duplicated) list is returned.

    Returns:
        Up to ``k`` distinct ids in rank order.

    Raises:
        ValueError: If ``k < 1``.
    """
    _validate_k(k)
    return dedupe_preserving_order(retrieved_ids)[:k]


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of relevant documents found within the top K.

    ``Recall@K = |top_k(retrieved) ∩ relevant| / |relevant|``

    Args:
        retrieved_ids: Retrieved ids in rank order.
        relevant_ids: Ground-truth relevant ids; order is irrelevant, duplicates
            are collapsed.
        k: Cut-off rank; must be >= 1.

    Returns:
        A value in ``[0.0, 1.0]``, or ``None`` when ``relevant_ids`` is empty.

        ``None`` is returned because the denominator ``|relevant|`` is zero:
        "what fraction of nothing did you find?" has no answer. Returning 0.0
        would punish a system for a question with no evidence, and 1.0 would
        reward every system equally. The harness surfaces this as ``SKIPPED``.
        Empty ``retrieved_ids`` is *not* undefined -- nothing was found, which
        is genuinely 0.0.

    Raises:
        ValueError: If ``k < 1``.
    """
    _validate_k(k)
    retrieved_ids = validated_ids(retrieved_ids)
    relevant = set(validated_ids(relevant_ids))
    if not relevant:
        return None
    hits = len(set(top_k(retrieved_ids, k)) & relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
    denominator: PrecisionDenominator = "retrieved",
) -> float | None:
    """Fraction of the top K results that are relevant.

    Two conventions exist and they disagree whenever the retriever returns fewer
    than K results. Both are implemented; the choice is explicit rather than
    hidden inside the formula.

    ``denominator="retrieved"`` (default)
        ``|top_k ∩ relevant| / |top_k|`` -- divide by how many distinct results
        were actually returned. This measures *precision of what was shown to
        the user*. A retriever that returns 2 results, both relevant, scores
        1.0 at K=5. This is the default because dividing by ``k`` conflates two
        different faults: "returned fewer results" and "returned wrong results".
        Under this convention the metric is **undefined when nothing was
        retrieved** (``None`` -> ``SKIPPED``).

    ``denominator="k"``
        ``|top_k ∩ relevant| / k`` -- divide by the fixed cut-off. This treats a
        short result list as a partial failure: the same retriever scores 0.4 at
        K=5. Always defined, including for an empty result list (0.0). Choose
        this when "filled all K slots" is itself a requirement, or when you want
        empty retrieval to score zero rather than skip.

    Args:
        retrieved_ids: Retrieved ids in rank order.
        relevant_ids: Ground-truth relevant ids. May be empty -- see Returns.
        k: Cut-off rank; must be >= 1.
        denominator: Which convention to apply.

    Returns:
        A value in ``[0.0, 1.0]``, or ``None`` when ``denominator="retrieved"``
        and the de-duplicated result list is empty.

        Unlike Recall@K, precision remains **defined when ``relevant_ids`` is
        empty**: the numerator is legitimately zero, and "you returned documents
        when none were relevant" is a real, meaningful score of 0.0. The one
        genuinely undefined combination -- nothing relevant *and* nothing
        retrieved, under the ``"retrieved"`` convention -- is caught by the
        empty-result check and reported as ``SKIPPED`` rather than scored 0.0,
        which would wrongly penalise a system that correctly returned nothing.

    Raises:
        ValueError: If ``k < 1``.
    """
    if denominator not in ("retrieved", "k"):
        raise ValueError("denominator must be retrieved or k")
    effective = top_k(retrieved_ids, k)
    hits = len(set(effective) & set(validated_ids(relevant_ids)))
    if denominator == "k":
        return hits / k
    if not effective:
        return None
    return hits / len(effective)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int | None = None,
) -> float | None:
    """Reciprocal of the rank of the first relevant document.

    ``RR = 1 / rank_of_first_relevant``, or ``0.0`` if no relevant document
    appears. Ranks are 1-based and computed *after* duplicate collapsing, so a
    repeated id cannot push a later document's rank down.

    This is the **per-case** quantity. Mean Reciprocal Rank is the mean of these
    across a suite -- see :func:`mean_reciprocal_rank`. The two are named
    separately on purpose: reporting a single case's RR as "MRR" is a common and
    misleading error.

    Args:
        retrieved_ids: Retrieved ids in rank order.
        relevant_ids: Ground-truth relevant ids.
        k: Optional cut-off. If given, only the first ``k`` distinct results are
            considered and a relevant document below K scores 0.0. If ``None``
            (the default), the entire list is searched.

    Returns:
        ``1/rank`` in ``(0.0, 1.0]`` if a relevant document is found, ``0.0`` if
        none is found, or ``None`` when ``relevant_ids`` is empty (no rank can
        be sought, so the metric is undefined -> ``SKIPPED``).

    Raises:
        ValueError: If ``k`` is given and is < 1.
    """
    if k is not None:
        _validate_k(k)
    retrieved_ids = validated_ids(retrieved_ids)
    relevant = set(validated_ids(relevant_ids))
    if not relevant:
        return None
    candidates = (
        top_k(retrieved_ids, k) if k is not None else dedupe_preserving_order(retrieved_ids)
    )
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float | None:
    """Mean Reciprocal Rank: the mean of per-case reciprocal ranks.

    Args:
        reciprocal_ranks: Per-case values from :func:`reciprocal_rank`. Callers
            must exclude cases where ``reciprocal_rank`` returned ``None``;
            those cases have no defined rank and must not be averaged in as 0.0.

    Returns:
        The mean, or ``None`` if no values were supplied -- MRR over zero cases
        is undefined, not zero.
    """
    return mean(reciprocal_ranks)
