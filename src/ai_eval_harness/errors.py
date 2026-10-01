"""Exception hierarchy for the harness.

Every failure mode the harness can produce is represented by a distinct exception
type so that callers (and the CLI) can map failures onto meaningful exit codes
instead of inspecting message strings.

The hierarchy intentionally separates three kinds of problem:

* :class:`ConfigError` -- the user asked for something the harness cannot do
  (unknown evaluator, threshold out of range, unreadable configuration).
* :class:`DatasetError` -- the input data is malformed or unusable.
* :class:`EvaluationError` -- something went wrong while evaluating.

Nothing in this module swallows an error; these types exist so that errors can be
*reported precisely*, never converted into a plausible-looking score.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "AiEvalHarnessError",
    "ConfigError",
    "DatasetError",
    "DatasetValidationError",
    "EmptyDatasetError",
    "EvaluationError",
    "EvaluatorExecutionError",
    "JudgeError",
    "JudgeResponseError",
    "JudgeUnavailableError",
    "MissingCredentialError",
    "MissingProviderDependencyError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "UnknownEvaluatorError",
]


class AiEvalHarnessError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ConfigError(AiEvalHarnessError):
    """The suite configuration is invalid or cannot be satisfied."""


class UnknownEvaluatorError(ConfigError):
    """A configuration referenced an evaluator type that is not registered."""

    def __init__(self, requested: str, available: Sequence[str]) -> None:
        self.requested = requested
        self.available = tuple(available)
        super().__init__(
            f"Unknown evaluator type {requested!r}. "
            f"Available types: {', '.join(self.available) or '(none registered)'}"
        )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class DatasetError(AiEvalHarnessError):
    """Base class for dataset loading problems."""


class DatasetValidationError(DatasetError):
    """One or more evaluation cases in a dataset are malformed.

    All problems found while loading a dataset are collected and reported
    together: a malformed case is never skipped silently.
    """

    def __init__(self, source: str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = tuple(problems)
        detail = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"{len(self.problems)} problem(s) found while loading dataset {source!r}:\n{detail}"
        )


class EmptyDatasetError(DatasetError):
    """The dataset contained no evaluation cases.

    Evaluating an empty dataset would produce a suite that passes every gate
    without measuring anything, so it is an error unless explicitly allowed.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(
            f"Dataset {source!r} contains no evaluation cases. "
            "An empty dataset cannot demonstrate quality; pass --allow-empty-dataset "
            "(or set allow_empty_dataset = true) if this is intentional."
        )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class EvaluationError(AiEvalHarnessError):
    """Base class for failures that occur while evaluating a case."""


class EvaluatorExecutionError(EvaluationError):
    """An evaluator raised while scoring a case.

    The runner catches this (and any other exception raised by an evaluator),
    records an ``ERROR`` result for the case, and continues. Errors are counted
    and surfaced in the report; they are never treated as a score of zero.
    """

    def __init__(self, evaluator: str, case_id: str, cause: BaseException) -> None:
        self.evaluator = evaluator
        self.case_id = case_id
        self.cause = cause
        super().__init__(
            f"Evaluator {evaluator!r} failed on case {case_id!r}: {type(cause).__name__}: {cause}"
        )


class JudgeError(EvaluationError):
    """Base class for LLM-judge failures."""


class JudgeUnavailableError(JudgeError):
    """An LLM judge was requested but no provider is able to serve it.

    Raised instead of returning a fabricated verdict. A missing judge must be
    visible in the report, not silently replaced by a heuristic.
    """


class JudgeResponseError(JudgeError):
    """A judge provider returned output that could not be parsed into a verdict."""

    def __init__(self, reason: str, raw_response: str) -> None:
        self.reason = reason
        self.raw_response = raw_response
        # Raw output is available only to an explicit programmatic caller; it must
        # never enter exception strings, CLI output, or persisted error reports.
        super().__init__(f"Could not parse judge response ({reason}).")


class MissingProviderDependencyError(ConfigError):
    """Install the optional SDK extra before constructing a live provider."""


class MissingCredentialError(ConfigError):
    """A live provider has no runtime credential."""


class ProviderError(JudgeError):
    """Sanitized provider failure; no original request, headers, or response body."""

    retryable = False


class ProviderAuthenticationError(ProviderError):
    """Authentication or permission failure; do not retry."""


class ProviderRateLimitError(ProviderError):
    """Rate limit; retry within the configured bound."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """Transport timeout; retry within the configured bound."""

    retryable = True


class ProviderConnectionError(ProviderError):
    """Connection failure; retry within the configured bound."""

    retryable = True


class ProviderServerError(ProviderError):
    """Server failure; retry within the configured bound."""

    retryable = True


class ProviderRequestError(ProviderError):
    """Permanent invalid request, unsupported model, or unexpected SDK failure."""
