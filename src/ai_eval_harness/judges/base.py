"""The judge provider contract.

A provider is anything that can turn a :class:`JudgeRequest` into a
:class:`JudgeResponse`. That is the entire interface -- deliberately narrow, so
that adapting a new vendor requires no knowledge of metrics, cases, or reports.

The request carries a system prompt and a user prompt because every mainstream
chat completion API accepts that shape; it carries no vendor-specific fields
(no temperature, no tool definitions, no response-format directives). A provider
is free to set those itself, and is expected to configure for determinism where
the vendor supports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["JudgeProvider", "JudgeRequest", "JudgeResponse", "JudgeVerdict"]


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """A single judging request, independent of any provider.

    Attributes:
        criterion: What is being judged, e.g. ``"groundedness"``. Used to select
            recorded responses and to label results.
        case_id: The case under evaluation. Carried so providers can key caches
            or recordings by case without parsing prompts.
        system_prompt: Instructions defining the judge's role and its required
            output format.
        user_prompt: The material to judge (question, context, answer).
        metadata: Free-form detail for provider adapters and reports.
    """

    criterion: str
    case_id: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    """Raw output from a provider, before parsing.

    Attributes:
        text: The provider's completion text, exactly as returned. Kept raw so
            that a parse failure can report what was actually received.
        model: Identifier of the model that produced the text. Recorded in
            reports: a judge score is only interpretable alongside the model
            that produced it.
        metadata: Provider-specific detail (token usage, stop reason, ...).
    """

    text: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A parsed, normalised judgement.

    Attributes:
        score: Confidence or quality in ``[0.0, 1.0]``, validated on
            construction. A judge that cannot express its opinion on this scale
            must be adapted by its provider, not by loosening this contract.
        reasoning: The judge's stated justification. Required: an unexplained
            model score is not reviewable, and reviewability is the only defence
            against a miscalibrated judge.
        raw: The parsed payload as received, for auditing.
    """

    score: float
    reasoning: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Judge score must be within [0.0, 1.0], got {self.score}")
        if not self.reasoning.strip():
            raise ValueError("Judge verdict must include non-empty reasoning")


@runtime_checkable
class JudgeProvider(Protocol):
    """Anything able to answer a :class:`JudgeRequest`.

    Implementations must raise rather than return a fabricated response when
    they cannot serve a request. A judge that guesses is worse than no judge:
    it produces numbers that look like measurements.
    """

    @property
    def name(self) -> str:
        """Short identifier for the provider, e.g. ``"scripted"``. Appears in reports."""
        ...

    @property
    def model(self) -> str:
        """Identifier of the model used. Appears in reports."""
        ...

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Answer a judging request.

        Args:
            request: The request to serve.

        Returns:
            The provider's raw response.

        Raises:
            ai_eval_harness.errors.JudgeUnavailableError: If the request cannot
                be served at all.
        """
        ...
