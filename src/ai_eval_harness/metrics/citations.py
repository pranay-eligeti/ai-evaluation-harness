"""Citation metrics: does the answer attribute its claims to the right sources?

Two independent questions are measured, and conflating them hides real failures:

**Citation correctness** (:func:`citation_scores`) compares the cited ids against
a human-annotated set of ids the answer *should* have cited. It needs ground
truth and answers "did it cite the right sources?".

**Citation validity** (:func:`citation_validity`) checks that each cited id
actually exists in the context the system retrieved. It needs no ground truth
and answers "does this citation resolve to anything at all?" -- catching the
fabricated-reference failure mode, where a model invents a plausible source id.
A system can be perfectly valid and completely incorrect: citing a real
retrieved document that does not support the claim.

Citations are treated as an **unordered set** of ids. Repeating a citation is not
additional evidence, so duplicates are collapsed before scoring. Ids are compared
exactly, consistent with the retrieval metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_eval_harness.metrics.validation import validated_ids

__all__ = ["CitationScores", "citation_scores", "citation_validity"]


@dataclass(frozen=True, slots=True)
class CitationScores:
    """Precision, recall, and F1 for one case's citations.

    Attributes:
        precision: Of the ids cited, the fraction that should have been cited.
        recall: Of the ids that should have been cited, the fraction that were.
        f1: Harmonic mean of the two; the headline score.
        matched: Ids that were both expected and cited, sorted for stable reports.
        spurious: Ids cited but not expected, sorted.
        missing: Ids expected but not cited, sorted.
    """

    precision: float
    recall: float
    f1: float
    matched: tuple[str, ...]
    spurious: tuple[str, ...]
    missing: tuple[str, ...]


def citation_scores(
    expected_citations: Sequence[str],
    actual_citations: Sequence[str],
) -> CitationScores:
    """Compare cited source ids against the ids that should have been cited.

    Boundary conventions, chosen so that each of the four empty/non-empty
    combinations expresses the right thing:

    ======================  ====================  =========  ======  ====
    expected                actual                precision  recall  F1
    ======================  ====================  =========  ======  ====
    ``[]``                  ``[]``                1.0        1.0     1.0
    ``[]``                  ``["d1"]``            0.0        1.0     0.0
    ``["d1"]``              ``[]``                1.0        0.0     0.0
    ``["d1"]``              ``["d1"]``            1.0        1.0     1.0
    ======================  ====================  =========  ======  ====

    The rule is: an empty set is *vacuously satisfied*. Precision over zero
    citations is 1.0 (none of them are wrong), recall over zero expectations is
    1.0 (all zero were found). F1 then collapses to 0.0 in exactly the two cases
    where the system got it wrong -- citing when it should not have, and failing
    to cite when it should have -- and to 1.0 when both are empty, which is the
    correct behaviour for an answer that legitimately has nothing to cite.
    Defining precision as 0.0 for an empty citation list would make "cited
    nothing" indistinguishable from "cited entirely wrongly".

    Args:
        expected_citations: Ground-truth ids. Must not be ``None``; a case with
            no citation annotation is ``SKIPPED`` by the evaluator before
            reaching this function, since ``[]`` here means the distinct and
            meaningful claim "this answer should cite nothing".
        actual_citations: Ids the system cited.

    Returns:
        A :class:`CitationScores`. Always defined -- never ``None``.
    """
    expected = set(validated_ids(expected_citations))
    actual = set(validated_ids(actual_citations))
    matched = expected & actual

    precision = 1.0 if not actual else len(matched) / len(actual)
    recall = 1.0 if not expected else len(matched) / len(expected)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return CitationScores(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=tuple(sorted(matched)),
        spurious=tuple(sorted(actual - expected)),
        missing=tuple(sorted(expected - actual)),
    )


def citation_validity(
    actual_citations: Sequence[str],
    retrieved_ids: Sequence[str],
) -> float | None:
    """Fraction of cited ids that resolve to a document the system retrieved.

    ``validity = |distinct cited ids ∩ retrieved ids| / |distinct cited ids|``

    A score below 1.0 means the answer referenced a source that was not in its
    own context -- either a hallucinated identifier or a citation to something
    the system never actually read. Either way the citation is unverifiable.

    Args:
        actual_citations: Ids the system cited.
        retrieved_ids: Ids of the documents in the system's context. Duplicates
            are irrelevant here; only membership matters.

    Returns:
        A value in ``[0.0, 1.0]``, or ``None`` when ``actual_citations`` is
        empty.

        ``None`` rather than ``1.0``: an answer with no citations has no invalid
        citations, but scoring that as perfect validity would reward omitting
        citations entirely. The metric is undefined (``SKIPPED``), and whether
        the answer *should* have cited something is the separate question
        measured by :func:`citation_scores`.
    """
    retrieved = set(validated_ids(retrieved_ids))
    cited = set(validated_ids(actual_citations))
    if not cited:
        return None
    return len(cited & retrieved) / len(cited)
