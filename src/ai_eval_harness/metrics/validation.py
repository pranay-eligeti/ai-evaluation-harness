"""Validation at the public pure-metric boundary."""

from collections.abc import Iterable
from math import isfinite


def validated_ids(values: Iterable[str]) -> list[str]:
    if isinstance(values, str | bytes):
        raise ValueError("IDs must be an iterable of identifiers, not a string")
    items = list(values)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError("Every identifier must be a nonblank string")
    return items


def validate_score(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Score must be numeric")
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Score must be finite and within [0, 1]")
