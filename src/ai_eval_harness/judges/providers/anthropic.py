"""Anthropic Messages API adapter with native JSON schema output."""

from typing import Any

from ai_eval_harness.errors import JudgeResponseError
from ai_eval_harness.judges.base import JudgeRequest, JudgeResponse
from ai_eval_harness.judges.providers.common import LiveJudgeProvider, verdict_schema


class AnthropicJudgeProvider(LiveJudgeProvider):
    @property
    def name(self) -> str:
        return "anthropic"

    def _send(self, request: JudgeRequest) -> Any:
        return self._client.messages.create(
            model=self.model,
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.user_prompt}],
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            output_config={"format": {"type": "json_schema", "schema": verdict_schema()}},
        )

    def _extract(self, response: Any) -> JudgeResponse:
        if getattr(response, "stop_reason", None) != "end_turn":
            raise JudgeResponseError("Anthropic response was refused, truncated, or incomplete", "")
        content = getattr(response, "content", None)
        model = getattr(response, "model", None)
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(model, str)
            or not model
        ):
            raise JudgeResponseError("Anthropic response has unexpected blocks or no model", "")
        block = content[0]
        text = getattr(block, "text", None)
        if getattr(block, "type", None) != "text" or not isinstance(text, str) or not text.strip():
            raise JudgeResponseError("Anthropic response does not contain verdict text", "")
        return JudgeResponse(text, model)
