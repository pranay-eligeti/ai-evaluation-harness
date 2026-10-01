"""OpenAI Responses API adapter; no SDK import at module load time."""

from typing import Any

from ai_eval_harness.errors import JudgeResponseError
from ai_eval_harness.judges.base import JudgeRequest, JudgeResponse
from ai_eval_harness.judges.providers.common import LiveJudgeProvider, verdict_schema


class OpenAIJudgeProvider(LiveJudgeProvider):
    @property
    def name(self) -> str:
        return "openai"

    def _send(self, request: JudgeRequest) -> Any:
        return self._client.responses.create(
            model=self.model,
            instructions=request.system_prompt,
            input=request.user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "judge_verdict",
                    "strict": True,
                    "schema": verdict_schema(),
                }
            },
            max_output_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            store=False,
        )

    def _extract(self, response: Any) -> JudgeResponse:
        if getattr(response, "status", None) != "completed":
            raise JudgeResponseError("OpenAI response was incomplete or failed", "")
        text = getattr(response, "output_text", None)
        model = getattr(response, "model", None)
        if not isinstance(text, str) or not text.strip() or not isinstance(model, str) or not model:
            raise JudgeResponseError("OpenAI response lacks verdict text or model provenance", "")
        return JudgeResponse(text, model)
