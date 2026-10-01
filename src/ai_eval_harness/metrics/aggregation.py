"""Aggregation helpers shared by the metrics and the runner.

Kept in one place so that "the mean of an empty sequence is ``None``, not ``0.0``"
is decided exactly once. Averaging nothing into zero is the single easiest way to
make a broken evaluation look like a failing system.
"""

from __future__ import annotations

from collections.abc import Iterable

from ai_eval_harness.metrics.validation import validate_score

__all__ = ["mean"]


def mean(values: Iterable[float]) -> float | None:
    """Arithmetic mean of ``values``.

    Args:
        values: The values to average.

    Returns:
        The mean, or ``None`` if ``values`` is empty. ``None`` propagates the
        fact that there was nothing to measure; callers must decide what that
        means rather than inheriting a fabricated ``0.0``.
    """
    materialised = list(values)
    for value in materialised:
        validate_score(value)
    if not materialised:
        return None
    return sum(materialised) / len(materialised)
