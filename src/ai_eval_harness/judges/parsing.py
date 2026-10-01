"""Parsing provider output into a :class:`~ai_eval_harness.judges.base.JudgeVerdict`.

Parsing is strict and total: either the response is a well-formed verdict, or
:class:`~ai_eval_harness.errors.JudgeResponseError` is raised with the raw text
attached. There is no salvage path -- no regex hunting for a number in prose, no
"assume 0.5 if unclear". A judge whose output cannot be parsed has not produced
an evaluation, and the harness reports that as an ``ERROR``, which is visible,
rather than a score, which would be fiction.

The expected payload is a single JSON object::

    {"score": 0.0-1.0, "reasoning": "<non-empty explanation>"}

``score`` may also be given as a boolean (``true``/``false`` -> ``1.0``/``0.0``)
because a pass/fail judge is a common and legitimate design. Extra keys are
preserved in :attr:`JudgeVerdict.raw` rather than rejected: providers routinely
add useful detail, and discarding it would lose audit information.

Many chat models wrap JSON in a Markdown fence. That one specific, unambiguous
wrapper is stripped before parsing; nothing else is repaired.
"""

from __future__ import annotations

import json
from typing import Any

from ai_eval_harness.errors import JudgeResponseError
from ai_eval_harness.judges.base import JudgeVerdict

__all__ = ["parse_verdict"]

_FENCE = "```"


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding Markdown code fence, if present."""
    stripped = text.strip()
    if not (stripped.startswith(_FENCE) and stripped.endswith(_FENCE) and len(stripped) > 6):
        return stripped
    inner = stripped[len(_FENCE) : -len(_FENCE)]
    # Drop an optional language tag on the opening fence, e.g. ```json
    first_newline = inner.find("\n")
    if first_newline != -1 and inner[:first_newline].strip().isalpha():
        inner = inner[first_newline + 1 :]
    return inner.strip()


def _coerce_score(value: Any) -> float:
    """Convert a payload ``score`` into a float in ``[0.0, 1.0]``.

    Raises:
        JudgeResponseError: If the value is not a number or boolean.
    """
    if isinstance(value, bool):
        # Checked before int: bool is a subclass of int in Python.
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    raise JudgeResponseError(
        f"'score' must be a number or boolean, got {type(value).__name__}", repr(value)
    )


def parse_verdict(response_text: str) -> JudgeVerdict:
    """Parse a judge's raw completion into a verdict.

    Args:
        response_text: The provider's completion text.

    Returns:
        The parsed verdict.

    Raises:
        ai_eval_harness.errors.JudgeResponseError: If the text is not a JSON
            object, is missing ``score`` or ``reasoning``, has a ``score`` of the
            wrong type or outside ``[0.0, 1.0]``, or has empty ``reasoning``.
    """
    candidate = _strip_code_fence(response_text)
    if not candidate:
        raise JudgeResponseError("response was empty", response_text)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"response is not valid JSON: {exc.msg}", response_text) from exc

    if not isinstance(payload, dict):
        raise JudgeResponseError(
            f"expected a JSON object, got {type(payload).__name__}", response_text
        )

    missing = [key for key in ("score", "reasoning") if key not in payload]
    if missing:
        raise JudgeResponseError(f"missing required key(s): {', '.join(missing)}", response_text)

    score = _coerce_score(payload["score"])
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str):
        raise JudgeResponseError(
            f"'reasoning' must be a string, got {type(reasoning).__name__}", response_text
        )

    try:
        # JudgeVerdict enforces the [0, 1] range and non-empty reasoning.
        return JudgeVerdict(score=score, reasoning=reasoning, raw=payload)
    except ValueError as exc:
        raise JudgeResponseError(str(exc), response_text) from exc
