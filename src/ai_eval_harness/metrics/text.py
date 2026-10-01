"""Text comparison metrics and the lexical groundedness proxy.

These metrics operate on surface forms. They cannot recognise a paraphrase, and
they are not a substitute for semantic evaluation -- see ``docs/METRICS.md`` and
``docs/LLM_JUDGES.md``. They are included because they are *reproducible*: they
give a stable regression signal that runs in CI with no credentials, which a
model-based judge cannot.

Normalisation
-------------

All comparisons run over a normalised form (:func:`normalize_text`), applied in
this fixed order:

1. Lowercase.
2. Replace every ASCII punctuation character with a space.
3. Drop the English articles ``a``, ``an``, ``the``.
4. Collapse runs of whitespace and strip.

Step 2 deviates deliberately from the original SQuAD normalisation, which
*deletes* punctuation. Deleting turns ``state-of-the-art`` into ``stateoftheart``
and ``10,000`` into ``10000``; replacing with a space yields ``state of art``
(after article removal) and ``10 000``. Splitting on punctuation matches the
token boundaries a reader would expect and keeps hyphenated compounds
comparable with their spaced spellings. The trade-off -- ``10,000`` no longer
matches ``10000`` -- is accepted and documented rather than left implicit.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = [
    "ARTICLES",
    "DEFAULT_STOPWORDS",
    "exact_match",
    "lexical_groundedness",
    "normalize_text",
    "token_f1",
    "tokenize",
]

ARTICLES = frozenset({"a", "an", "the"})
"""English articles removed during normalisation, following SQuAD convention."""

_PUNCTUATION_TO_SPACE = str.maketrans(dict.fromkeys(string.punctuation, " "))

DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)
"""Function words excluded from the groundedness proxy.

Defined explicitly, as a module-level constant, so the metric has no hidden
magic list: if this set changes, groundedness scores change, and the change is
reviewable in a diff. It is a small closed-class English list only -- it carries
no domain assumptions.
"""


def normalize_text(text: str) -> str:
    """Normalise text for surface comparison. See the module docstring for the rules.

    Args:
        text: Raw text.

    Returns:
        The normalised string, possibly empty.
    """
    lowered = text.lower().translate(_PUNCTUATION_TO_SPACE)
    return " ".join(token for token in lowered.split() if token not in ARTICLES)


def tokenize(text: str) -> list[str]:
    """Split text into normalised whitespace-delimited tokens.

    Args:
        text: Raw text.

    Returns:
        The tokens of :func:`normalize_text`, possibly an empty list.
    """
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, reference: str) -> float:
    """Whether two strings are identical after normalisation.

    Args:
        prediction: The generated answer.
        reference: The gold answer.

    Returns:
        ``1.0`` if the normalised forms are equal, otherwise ``0.0``.

        Two empty (or punctuation-only) strings normalise to ``""`` and match.
        Callers are responsible for deciding whether an empty answer should
        reach this function at all; the evaluators in this package treat a
        missing answer as ``SKIPPED`` and an empty answer as a real, scored
        response. Always defined -- never ``None``.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between a prediction and a reference.

    Tokens are compared as a **multiset**: if the reference says "very very
    large" and the prediction says "very large", one of the two ``very`` tokens
    counts as a match, not both. Using a plain set would let a prediction earn
    credit for repetition it never produced.

    ``precision = overlap / |prediction tokens|``,
    ``recall = overlap / |reference tokens|``,
    ``F1 = 2·precision·recall / (precision + recall)``.

    Args:
        prediction: The generated answer.
        reference: The gold answer.

    Returns:
        A value in ``[0.0, 1.0]``.

        If *both* normalise to zero tokens the result is ``1.0`` (two empty
        answers agree). If exactly one is empty the result is ``0.0``. If they
        share no tokens the result is ``0.0``. Always defined -- never ``None``.
    """
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        # Agreement on emptiness is a match; disagreement is a total miss.
        return 1.0 if not prediction_tokens and not reference_tokens else 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def lexical_groundedness(
    answer: str,
    context_texts: Sequence[str],
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> float | None:
    """Fraction of an answer's distinct content words that appear in the context.

    ``groundedness = |distinct content tokens of answer ∩ context tokens| /
    |distinct content tokens of answer|``

    .. warning::
       **This is a weak lexical proxy, not faithfulness.** It measures vocabulary
       overlap and nothing else. It cannot detect a claim that reuses the
       context's words while inverting their meaning ("the drug is *not*
       approved"), and it penalises a perfectly faithful answer that paraphrases.
       Treat a low score as a signal worth investigating and a high score as
       weak evidence only. Genuine faithfulness assessment requires a semantic
       judge -- see ``docs/LLM_JUDGES.md``.

    Content words are the distinct tokens remaining after normalisation once
    ``stopwords`` are removed. Distinct rather than multiset: repeating a word
    ten times should neither help nor hurt grounding.

    Args:
        answer: The generated answer.
        context_texts: Text of the retrieved documents. Documents without text
            must be omitted by the caller.
        stopwords: Function words to ignore. Defaults to :data:`DEFAULT_STOPWORDS`.

    Returns:
        A value in ``[0.0, 1.0]``, or ``None`` when the metric is undefined:

        * no ``context_texts`` were supplied -- there is nothing to be grounded
          in, so the answer's grounding cannot be assessed (an answer is not
          *ungrounded* merely because the harness was given no document text);
        * the answer contains no content words after stopword removal -- the
          denominator would be zero.

        Both are reported as ``SKIPPED``. Note the asymmetry with an *empty*
        answer, which also has no content words and is therefore also undefined
        here; answer *quality* for an empty answer is captured by
        :func:`token_f1`, which scores it 0.0.
    """
    if not context_texts:
        return None

    stopword_set = frozenset(stopwords)
    answer_content = {token for token in tokenize(answer) if token not in stopword_set}
    if not answer_content:
        return None

    context_tokens: set[str] = set()
    for text in context_texts:
        context_tokens.update(tokenize(text))

    return len(answer_content & context_tokens) / len(answer_content)
