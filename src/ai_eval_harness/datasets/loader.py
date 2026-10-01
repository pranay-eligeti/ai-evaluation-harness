"""Loading evaluation datasets from disk.

Supported formats:

``.jsonl``
    One JSON object per line. The preferred format: line-oriented files diff
    cleanly, stream, and let an error be reported against a specific line.
``.json``
    A JSON array of objects, or an object with a ``cases`` array and an optional
    ``name``.

Validation is **strict and exhaustive**. Every problem in the file is collected
and reported together, and a file with any problem loads nothing at all. Two
failure modes are deliberately excluded:

* *Skip the bad rows.* A dataset that quietly shrinks produces a suite that
  passes its gates on the cases that happened to parse. Coverage loss must never
  be invisible.
* *Stop at the first error.* Fixing a malformed dataset one error per run is
  needlessly slow when every error is already known.

Malformed input is never repaired. An unrecognised field is an error, not an
ignored key -- a misspelled ``expected_document_ids`` would otherwise silently
become "unannotated", turning a real measurement into a skip.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_eval_harness.errors import DatasetError, DatasetValidationError, EmptyDatasetError
from ai_eval_harness.models import EvaluationCase, EvaluationDataset

__all__ = ["load_dataset", "load_dataset_from_records", "parse_jsonl"]

_SUPPORTED_SUFFIXES = (".jsonl", ".json")


def _format_validation_error(location: str, error: ValidationError) -> list[str]:
    """Render a pydantic error into one readable message per invalid field."""
    messages: list[str] = []
    for detail in error.errors():
        field_path = ".".join(str(part) for part in detail["loc"]) or "(root)"
        messages.append(f"{location}: field {field_path!r}: {detail['msg']}")
    return messages


def parse_jsonl(text: str, source: str = "<string>") -> list[tuple[str, Any]]:
    """Parse JSON Lines text into located records.

    Blank lines (and whitespace-only lines) are ignored: they carry no data and
    are a common artefact of editors. Every other line must be valid JSON.

    Args:
        text: The file contents.
        source: Label used in error messages.

    Returns:
        ``(location, value)`` pairs, where ``location`` is e.g. ``"line 7"``.

    Raises:
        DatasetValidationError: If any non-blank line is not valid JSON. All
            offending lines are reported together.
    """
    records: list[tuple[str, Any]] = []
    problems: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        location = f"line {line_number}"
        try:
            records.append((location, json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"{location}: not valid JSON: {exc.msg}")
    if problems:
        raise DatasetValidationError(source, problems)
    return records


def _parse_json_document(text: str, source: str) -> tuple[list[tuple[str, Any]], str | None]:
    """Parse a ``.json`` dataset into located records plus an optional dataset name."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(source, [f"not valid JSON: {exc.msg}"]) from exc

    name: str | None = None
    if isinstance(document, Mapping):
        unknown = set(document) - {"name", "cases"}
        if unknown:
            raise DatasetValidationError(source, [f"Unknown dataset fields: {sorted(unknown)}"])
        if "cases" not in document:
            raise DatasetValidationError(
                source,
                ["a JSON object dataset must contain a 'cases' array"],
            )
        raw_name = document.get("name")
        if raw_name is not None and not isinstance(raw_name, str):
            raise DatasetValidationError(
                source, [f"'name' must be a string, got {type(raw_name).__name__}"]
            )
        name = raw_name
        entries = document["cases"]
    else:
        entries = document

    if not isinstance(entries, list):
        raise DatasetValidationError(
            source,
            [f"expected a JSON array of cases, got {type(entries).__name__}"],
        )
    return [(f"item {index}", entry) for index, entry in enumerate(entries)], name


def _build_cases(
    located_records: Sequence[tuple[str, Any]],
) -> tuple[list[EvaluationCase], list[str]]:
    """Validate located records into cases, collecting every problem found."""
    cases: list[EvaluationCase] = []
    problems: list[str] = []
    for location, record in located_records:
        if not isinstance(record, Mapping):
            problems.append(f"{location}: expected a JSON object, got {type(record).__name__}")
            continue
        try:
            cases.append(EvaluationCase.model_validate(dict(record)))
        except ValidationError as exc:
            problems.extend(_format_validation_error(location, exc))
    return cases, problems


def _assemble_dataset(
    *,
    name: str,
    cases: list[EvaluationCase],
    source: str,
    source_path: str | None,
    checksum: str | None,
    allow_empty: bool,
) -> EvaluationDataset:
    """Build the dataset model, converting dataset-level failures into harness errors."""
    if not cases and not allow_empty:
        raise EmptyDatasetError(source)
    try:
        return EvaluationDataset(name=name, cases=cases, source_path=source_path, checksum=checksum)
    except ValidationError as exc:
        raise DatasetValidationError(source, _format_validation_error("dataset", exc)) from exc


def load_dataset_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    name: str = "in-memory",
    allow_empty: bool = False,
) -> EvaluationDataset:
    """Build a dataset from already-parsed records.

    Useful for programmatic use and for tests, which should not have to write
    temporary files to exercise validation.

    Args:
        records: Case payloads.
        name: Dataset name recorded in reports.
        allow_empty: Permit a dataset with no cases.

    Returns:
        The validated dataset.

    Raises:
        DatasetValidationError: If any record is invalid, or case ids collide.
        EmptyDatasetError: If there are no cases and ``allow_empty`` is False.
    """
    located = [(f"item {index}", record) for index, record in enumerate(records)]
    cases, problems = _build_cases(located)
    if problems:
        raise DatasetValidationError(name, problems)
    return _assemble_dataset(
        name=name,
        cases=cases,
        source=name,
        source_path=None,
        checksum=None,
        allow_empty=allow_empty,
    )


def load_dataset(
    path: Path,
    *,
    name: str | None = None,
    allow_empty: bool = False,
) -> EvaluationDataset:
    """Load and validate a dataset from a ``.jsonl`` or ``.json`` file.

    Args:
        path: Path to the dataset file.
        name: Dataset name for reports. Defaults to the name embedded in a
            ``.json`` document, otherwise the file stem.
        allow_empty: Permit a dataset with no cases. Off by default: an empty
            dataset would satisfy every quality gate without measuring anything.

    Returns:
        The validated dataset, with ``source_path`` and a SHA-256 ``checksum``
        of the file's exact bytes recorded for provenance.

    Raises:
        DatasetError: If the file cannot be read or has an unsupported extension.
        DatasetValidationError: If the file is not valid JSON/JSONL, if any case
            is invalid, or if case ids collide. All problems are reported together.
        EmptyDatasetError: If the file contains no cases and ``allow_empty`` is False.
    """
    source = str(path)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise DatasetError(
            f"Unsupported dataset extension {suffix or '(none)'!r} for {source!r}. "
            f"Supported extensions: {', '.join(_SUPPORTED_SUFFIXES)}"
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"Could not read dataset {source!r}: {exc}") from exc

    checksum = hashlib.sha256(raw_bytes).hexdigest()
    try:
        # utf-8-sig tolerates a byte-order mark, which editors on Windows add.
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(source, [f"file is not valid UTF-8: {exc}"]) from exc

    embedded_name: str | None = None
    if suffix == ".jsonl":
        located = parse_jsonl(text, source=source)
    else:
        located, embedded_name = _parse_json_document(text, source=source)

    cases, problems = _build_cases(located)
    if problems:
        raise DatasetValidationError(source, problems)

    return _assemble_dataset(
        name=name or embedded_name or path.stem,
        cases=cases,
        source=source,
        source_path=source,
        checksum=checksum,
        allow_empty=allow_empty,
    )
