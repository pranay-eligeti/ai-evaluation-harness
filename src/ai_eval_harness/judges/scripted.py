"""Offline judge providers.

:class:`ScriptedJudgeProvider` replays verdicts recorded in a file. It exists so
the LLM-judge code path is *executed* -- prompts built, responses parsed, results
normalised, gates applied -- in tests and in CI, with no credentials and no
network. Without it the judge abstraction would be untested code that merely
looks plausible.

A scripted provider is a test double, not an evaluation. It proves the plumbing
works; it says nothing about how a real model would score. Reports label its
results ``llm_judge`` with model ``scripted:<name>`` so a recorded run can never
be mistaken for a live one.

:class:`NullJudgeProvider` is the default. It refuses every request, so a suite
that configures a judge without configuring a provider fails loudly instead of
quietly producing invented numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ai_eval_harness.errors import ConfigError, JudgeUnavailableError
from ai_eval_harness.judges.base import JudgeRequest, JudgeResponse

__all__ = ["NullJudgeProvider", "ScriptedJudgeProvider", "load_scripted_provider"]


class NullJudgeProvider:
    """A provider that always refuses.

    Used when no judge is configured. Every call raises
    :class:`~ai_eval_harness.errors.JudgeUnavailableError`, which the runner
    records as an ``ERROR`` result -- never as a score.
    """

    @property
    def name(self) -> str:
        return "none"

    @property
    def model(self) -> str:
        return "none"

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Always raise; see the class docstring.

        Raises:
            JudgeUnavailableError: Always.
        """
        raise JudgeUnavailableError(
            f"No judge provider is configured, so criterion {request.criterion!r} "
            f"cannot be evaluated for case {request.case_id!r}. Configure a provider "
            "under [judge] in the suite configuration, or remove the LLM-judge "
            "evaluator from the suite."
        )


class ScriptedJudgeProvider:
    """Replays pre-recorded verdicts keyed by ``(criterion, case_id)``.

    The recorded payloads are serialised to JSON text and returned as raw
    completion text, so :func:`~ai_eval_harness.judges.parsing.parse_verdict`
    runs for real rather than being bypassed.

    Args:
        responses: ``{criterion: {case_id: payload}}``. Each payload is a
            verdict object (``{"score": ..., "reasoning": ...}``) or a raw
            string to be returned verbatim -- the latter lets tests exercise
            malformed-response handling.
        model: Identifier recorded in reports. Prefixed with ``scripted:`` so a
            replayed run is never mistaken for a live model call.
        name: Provider name recorded in reports.

    Raises:
        JudgeUnavailableError: From :meth:`complete`, when no verdict was
            recorded for the requested criterion and case. A missing recording
            is an incomplete fixture, which must surface as an error rather than
            be filled in with a default score.
    """

    def __init__(
        self,
        responses: Mapping[str, Mapping[str, Any]],
        *,
        model: str = "scripted-judge-v1",
        name: str = "scripted",
    ) -> None:
        self._responses = {criterion: dict(by_case) for criterion, by_case in responses.items()}
        self._model = f"scripted:{model}"
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Return the recorded response for ``request``.

        Args:
            request: The judging request.

        Returns:
            The recorded payload as raw completion text.

        Raises:
            JudgeUnavailableError: If nothing was recorded for this criterion
                and case.
        """
        by_case = self._responses.get(request.criterion)
        if by_case is None or request.case_id not in by_case:
            raise JudgeUnavailableError(
                f"No recorded judge response for criterion {request.criterion!r} "
                f"and case {request.case_id!r}. The scripted fixture is incomplete; "
                "this case cannot be judged."
            )
        payload = by_case[request.case_id]
        text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        return JudgeResponse(
            text=text,
            model=self._model,
            metadata={"replayed": True, "criterion": request.criterion},
        )


def load_scripted_provider(path: Path) -> ScriptedJudgeProvider:
    """Load a :class:`ScriptedJudgeProvider` from a JSON file.

    Expected file shape::

        {
          "model": "scripted-judge-v1",
          "responses": {
            "groundedness": {"case-1": {"score": 1.0, "reasoning": "..."}}
          }
        }

    Args:
        path: Path to the recordings file.

    Returns:
        The configured provider.

    Raises:
        ai_eval_harness.errors.ConfigError: If the file is missing, is not valid
            JSON, or does not have the expected structure.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read judge responses file {str(path)!r}: {exc}") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Judge responses file {str(path)!r} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise ConfigError(
            f"Judge responses file {str(path)!r} must contain a JSON object, "
            f"got {type(document).__name__}"
        )

    responses = document.get("responses")
    if not isinstance(responses, dict):
        raise ConfigError(
            f"Judge responses file {str(path)!r} must contain a 'responses' object "
            "mapping criterion -> case_id -> verdict"
        )
    for criterion, by_case in responses.items():
        if not isinstance(by_case, dict):
            raise ConfigError(
                f"Judge responses file {str(path)!r}: entry for criterion "
                f"{criterion!r} must be an object mapping case_id -> verdict"
            )

    model = document.get("model", "scripted-judge-v1")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"Judge responses file {str(path)!r}: 'model' must be a non-empty string")

    return ScriptedJudgeProvider(responses, model=model)
