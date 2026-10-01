"""Optional SDK contract tests; HTTP is intercepted, never sent to providers."""

import json
from typing import Any, Literal

import pytest

from ai_eval_harness.errors import ProviderAuthenticationError, ProviderTimeoutError
from ai_eval_harness.judges.base import JudgeRequest
from ai_eval_harness.judges.providers import ProviderConfig, build_provider


def payload(name: str) -> dict[str, Any]:
    text = '{"score":1,"reasoning":"supported"}'
    if name == "openai":
        return {
            "id": "resp_offline",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "returned-model",
            "output": [
                {
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "returned-model",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


@pytest.mark.parametrize("name", ["openai", "anthropic"])
@pytest.mark.parametrize("outcome", ["success", "auth", "timeout"])
def test_real_sdk_offline(name: Literal["openai", "anthropic"], outcome: str) -> None:
    sdk = pytest.importorskip(name)
    http = pytest.importorskip("httpx2")
    requests: list[dict[str, Any]] = []

    def handle(request: Any) -> Any:
        requests.append(json.loads(request.content))
        assert request.url.host in ("api.openai.com", "api.anthropic.com")
        assert request.url.path.endswith("/responses" if name == "openai" else "/messages")
        if outcome == "timeout":
            raise http.ReadTimeout("runtime-secret", request=request)
        if outcome == "auth":
            return http.Response(
                401, json={"error": {"type": "authentication_error", "message": "runtime-secret"}}
            )
        return http.Response(200, json=payload(name))

    constructor = sdk.OpenAI if name == "openai" else sdk.Anthropic
    with (
        http.Client(transport=http.MockTransport(handle)) as transport,
        constructor(api_key="offline-placeholder", http_client=transport) as client,
    ):
        config = ProviderConfig(provider=name, model="requested-model", max_retries=0)
        provider = build_provider(config, client=client)
        request = JudgeRequest("groundedness", "a", "rubric", '{"answer":"blue"}')
        if outcome == "success":
            result = provider.complete(request)
            assert result.model == "returned-model"
            assert json.loads(result.text)["score"] == 1
        else:
            expected = ProviderTimeoutError if outcome == "timeout" else ProviderAuthenticationError
            with pytest.raises(expected) as error:
                provider.complete(request)
            assert "runtime-secret" not in str(error.value)
    assert len(requests) == 1
    assert requests[0]["model"] == "requested-model"
