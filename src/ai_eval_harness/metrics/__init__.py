"""Deterministic metric implementations.

Every function in this subpackage is pure: no I/O, no global state, no network,
no randomness. Given the same arguments it returns the same value, on any machine,
forever. That property is what lets the harness be used as a CI quality gate.

Two distinct failure mechanisms are used deliberately and consistently:

* **``ValueError``** signals a *programmer* error -- an argument that can never be
  valid, such as ``k=0``. This is a bug in the caller's configuration and is
  surfaced loudly.
* **``None``** signals that the metric is *mathematically undefined for this data*
  -- for example Recall@K when no document is relevant. This is a legitimate
  state of the world; the harness reports it as ``SKIPPED``.

Neither is ever converted into ``0.0``. The exact conditions under which each
metric returns ``None`` are documented per function and in ``docs/METRICS.md``.
"""

from ai_eval_harness.metrics.aggregation import mean
from ai_eval_harness.metrics.citations import (
    CitationScores,
    citation_scores,
    citation_validity,
)
from ai_eval_harness.metrics.retrieval import (
    PrecisionDenominator,
    dedupe_preserving_order,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    top_k,
)
from ai_eval_harness.metrics.text import (
    ARTICLES,
    DEFAULT_STOPWORDS,
    exact_match,
    lexical_groundedness,
    normalize_text,
    token_f1,
    tokenize,
)

__all__ = [
    "ARTICLES",
    "DEFAULT_STOPWORDS",
    "CitationScores",
    "PrecisionDenominator",
    "citation_scores",
    "citation_validity",
    "dedupe_preserving_order",
    "exact_match",
    "lexical_groundedness",
    "mean",
    "mean_reciprocal_rank",
    "normalize_text",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "token_f1",
    "tokenize",
    "top_k",
]
