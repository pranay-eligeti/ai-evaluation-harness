"""Every provider branch runs offline using injected SDK-shaped clients."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from ai_eval_harness.errors import (
    JudgeResponseError,
    MissingCredentialError,
    MissingProviderDependencyError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)
from ai_eval_harness.judges.base import JudgeRequest
from ai_eval_harness.judges.providers import (
    AnthropicJudgeProvider,
    OpenAIJudgeProvider,
    ProviderConfig,
    build_provider,
)
from ai_eval_harness.judges.providers.common import LiveJudgeProvider

REQUEST = JudgeRequest("answer_relevance", "a", "system rubric", '{"answer":"blue"}')
ADAPTERS = [("openai", OpenAIJudgeProvider), ("anthropic", AnthropicJudgeProvider)]


def response(provider: str, text: str = '{"score":0.75,"reasoning":"partial"}') -> Any:
    if provider == "openai":
        return SimpleNamespace(status="completed", output_text=text, model="returned-model")
    return SimpleNamespace(
        stop_reason="end_turn",
        model="returned-model",
        content=[SimpleNamespace(type="text", text=text)],
    )


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.options: dict[str, Any] = {}
        self.responses = self
        self.messages = self

    def with_options(self, **kwargs: Any) -> "FakeClient":
        self.options = kwargs
        return self

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StatusError(Exception):
    def __init__(self, status: int) -> None:
        self.status_code = status
        super().__init__("private-header runtime-secret evaluated-content")


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
def test_request_and_success(name: str, adapter: type[LiveJudgeProvider]) -> None:
    client = FakeClient([response(name)])
    config = ProviderConfig(provider=name, model="requested-model", timeout=7.0)  # type: ignore[arg-type]
    result = adapter(config, client=client).complete(REQUEST)
    assert result.model == "returned-model"
    assert json.loads(result.text)["score"] == 0.75
    assert result.metadata["attempts"] == 1
    assert client.options == {"max_retries": 0, "timeout": 7.0}
    sent = client.requests[0]
    assert sent["model"] == "requested-model" and sent["timeout"] == 7.0
    if name == "openai":
        assert sent["instructions"] == REQUEST.system_prompt
        assert sent["input"] == REQUEST.user_prompt
        assert sent["store"] is False
        contract = sent["text"]["format"]
        assert contract["strict"] is True
    else:
        assert sent["system"] == REQUEST.system_prompt
        assert sent["messages"] == [{"role": "user", "content": REQUEST.user_prompt}]
        contract = sent["output_config"]["format"]
    assert contract["type"] == "json_schema"
    assert contract["schema"]["additionalProperties"] is False
    assert set(contract["schema"]["required"]) == {"score", "reasoning"}


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
@pytest.mark.parametrize(
    "payload",
    [
        "not-json runtime-secret",
        "[]",
        "{}",
        '{"score":true,"reasoning":"x"}',
        '{"score":1,"reasoning":" "}',
        '{"score":NaN,"reasoning":"x"}',
        '{"score":Infinity,"reasoning":"x"}',
        '{"score":-1,"reasoning":"x"}',
        '{"score":"1","reasoning":"x"}',
        '{"score":1,"reasoning":"x","unexpected":1}',
    ],
)
def test_malformed_not_retried(name: str, adapter: type[LiveJudgeProvider], payload: str) -> None:
    client = FakeClient([response(name, payload)])
    config = ProviderConfig(provider=name, model="model", max_retries=3)  # type: ignore[arg-type]
    with pytest.raises(JudgeResponseError) as error:
        adapter(config, client=client).complete(REQUEST)
    assert len(client.requests) == 1
    assert "runtime-secret" not in str(error.value)


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
@pytest.mark.parametrize(
    ("failure", "expected", "retryable"),
    [
        (StatusError(401), ProviderAuthenticationError, False),
        (StatusError(403), ProviderAuthenticationError, False),
        (StatusError(429), ProviderRateLimitError, True),
        (StatusError(500), ProviderServerError, True),
        (StatusError(400), ProviderRequestError, False),
        (TimeoutError("runtime-secret"), ProviderTimeoutError, True),
        (ConnectionError("runtime-secret"), ProviderConnectionError, True),
        (RuntimeError("runtime-secret"), ProviderRequestError, False),
    ],
)
def test_errors_sanitized_and_bounded(
    name: str,
    adapter: type[LiveJudgeProvider],
    failure: Exception,
    expected: type[Exception],
    retryable: bool,
) -> None:
    client = FakeClient([failure, failure, failure])
    sleeps: list[float] = []
    config = ProviderConfig(provider=name, model="model", max_retries=2, retry_delay=0.1)  # type: ignore[arg-type]
    with pytest.raises(expected) as error:
        adapter(config, client=client, sleep=sleeps.append).complete(REQUEST)
    assert len(client.requests) == (3 if retryable else 1)
    assert sleeps == ([0.1, 0.1] if retryable else [])
    assert "runtime-secret" not in str(error.value)


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
def test_retry_success(name: str, adapter: type[LiveJudgeProvider]) -> None:
    client = FakeClient([StatusError(429), response(name)])
    config = ProviderConfig(provider=name, model="model", retry_delay=0.0)  # type: ignore[arg-type]
    result = adapter(config, client=client, sleep=lambda _: None).complete(REQUEST)
    assert result.metadata["attempts"] == 2


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
def test_dependency_and_credentials(
    name: str, adapter: type[LiveJudgeProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ProviderConfig(provider=name, model="model")  # type: ignore[arg-type]

    def unavailable(module: str) -> Any:
        raise ImportError("private path")

    monkeypatch.setattr("ai_eval_harness.judges.providers.common.load_sdk", unavailable)
    with pytest.raises(MissingProviderDependencyError, match=f"\\[{name}\\]"):
        adapter(config)
    monkeypatch.setattr(
        "ai_eval_harness.judges.providers.common.load_sdk",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.delenv(f"{name.upper()}_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match=f"{name.upper()}_API_KEY"):
        adapter(config)


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
def test_constructor_and_close(
    name: str, adapter: type[LiveJudgeProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}
    closed: list[bool] = []

    def constructor(**kwargs: Any) -> Any:
        received.update(kwargs)
        return SimpleNamespace(close=lambda: closed.append(True))

    sdk = SimpleNamespace(OpenAI=constructor, Anthropic=constructor)
    monkeypatch.setattr("ai_eval_harness.judges.providers.common.load_sdk", lambda _: sdk)
    monkeypatch.setenv(f"{name.upper()}_API_KEY", "runtime-placeholder")
    config = ProviderConfig(provider=name, model="model")  # type: ignore[arg-type]
    with adapter(config):
        pass
    assert received["api_key"] == "runtime-placeholder"
    assert received["max_retries"] == 0
    assert received["base_url"].startswith("https://api.")
    assert closed == [True]
    assert "runtime-placeholder" not in config.model_dump_json()


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
def test_incomplete_or_unexpected_response(name: str, adapter: type[LiveJudgeProvider]) -> None:
    bad = response(name)
    if name == "openai":
        bad.status = "incomplete"
    else:
        bad.stop_reason = "max_tokens"
    client = FakeClient([bad])
    config = ProviderConfig(provider=name, model="model")  # type: ignore[arg-type]
    with pytest.raises(JudgeResponseError):
        adapter(config, client=client).complete(REQUEST)


@pytest.mark.parametrize(
    "updates",
    [
        {"provider": "unknown"},
        {"model": " "},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_retries": 4},
        {"max_retries": True},
        {"retry_delay": -1},
        {"prompt_version": "1"},
        {"api_key": "runtime-placeholder"},
    ],
)
def test_invalid_configuration(updates: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate({"provider": "openai", "model": "m", **updates})


def test_factory() -> None:
    for name, adapter in ADAPTERS:
        config = ProviderConfig(provider=name, model="model")  # type: ignore[arg-type]
        assert isinstance(build_provider(config, client=FakeClient([])), adapter)


@pytest.mark.parametrize(("name", "adapter"), ADAPTERS)
@pytest.mark.parametrize("problem", ["model", "text", "blocks"])
def test_response_shape_failures(name: str, adapter: type[LiveJudgeProvider], problem: str) -> None:
    bad = response(name)
    if problem == "model":
        bad.model = None
    elif name == "openai":
        bad.output_text = None if problem == "text" else ""
    elif problem == "text":
        bad.content[0].text = None
    else:
        bad.content.append(SimpleNamespace(type="tool_use", input={}))
    config = ProviderConfig(provider=name, model="model")  # type: ignore[arg-type]
    with pytest.raises(JudgeResponseError):
        adapter(config, client=FakeClient([bad])).complete(REQUEST)
