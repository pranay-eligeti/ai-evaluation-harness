"""Small shared transport policy. Optional SDKs are imported only at construction."""

import importlib
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self

from ai_eval_harness.errors import (
    ConfigError,
    MissingCredentialError,
    MissingProviderDependencyError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)
from ai_eval_harness.judges.base import JudgeRequest, JudgeResponse
from ai_eval_harness.judges.parsing import parse_verdict
from ai_eval_harness.judges.providers.config import ProviderConfig


def load_sdk(name: str) -> Any:
    return importlib.import_module(name)


def verdict_schema() -> dict[str, Any]:
    # A portable schema subset accepted by both vendors; bounds/nonblank reasoning
    # are still enforced locally even when the service reports schema conformance.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "number", "description": "Quality score from 0 to 1"},
            "reasoning": {"type": "string", "description": "Nonblank justification"},
        },
        "required": ["score", "reasoning"],
    }


def normalize_error(exc: Exception, provider: str) -> ProviderError:
    # Never interpolate str(exc): SDK exceptions can contain headers, credentials,
    # request content, and raw API error bodies. Match documented exception families.
    classes = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if classes & {"APITimeoutError", "TimeoutError"}:
        return ProviderTimeoutError(f"{provider}: request timed out")
    if (
        status in (401, 403)
        or "AuthenticationError" in classes
        or "PermissionDeniedError" in classes
    ):
        return ProviderAuthenticationError(f"{provider}: authentication or permission denied")
    if status == 429 or "RateLimitError" in classes:
        return ProviderRateLimitError(f"{provider}: rate limit exceeded")
    if isinstance(status, int) and status >= 500:
        return ProviderServerError(f"{provider}: provider server failure")
    if "APIConnectionError" in classes or isinstance(exc, ConnectionError):
        return ProviderConnectionError(f"{provider}: connection failed")
    return ProviderRequestError(f"{provider}: request rejected or unexpected SDK failure")


class LiveJudgeProvider(ABC):
    """SDK retries disabled; one bounded retry policy, injected sleeper for tests."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any = None,
        api_key: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.provider != self.name:
            raise ConfigError("Provider configuration does not match adapter")
        self.config = config
        self._sleep = sleep
        self._owns_client = client is None
        if client is not None:
            try:
                self._client = client.with_options(max_retries=0, timeout=config.timeout)
            except Exception:
                raise ProviderRequestError(f"{self.name}: invalid injected SDK client") from None
            return
        module_name = self.name
        try:
            sdk = load_sdk(module_name)
        except ImportError:
            raise MissingProviderDependencyError(
                f"Install ai-evaluation-harness[{module_name}] to use this provider"
            ) from None
        environment = f"{module_name.upper()}_API_KEY"
        credential = api_key if api_key is not None else os.environ.get(environment)
        if not credential or not credential.strip():
            raise MissingCredentialError(f"Set {environment} before live evaluation")
        try:
            constructor = sdk.OpenAI if self.name == "openai" else sdk.Anthropic
            # Fixed official endpoints prevent SDK environment overrides redirecting
            # evaluation data/credentials to an arbitrary service.
            endpoint = (
                "https://api.openai.com/v1"
                if self.name == "openai"
                else "https://api.anthropic.com"
            )
            self._client = constructor(
                api_key=credential, timeout=config.timeout, max_retries=0, base_url=endpoint
            )
        except Exception:
            raise ProviderRequestError(f"{self.name}: could not initialize SDK client") from None

    @property
    def model(self) -> str:
        return self.config.model

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _send(self, request: JudgeRequest) -> Any: ...

    @abstractmethod
    def _extract(self, response: Any) -> JudgeResponse: ...

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        response: Any = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._send(request)
            except Exception as exc:
                failure = normalize_error(exc, self.name)
                if not failure.retryable or attempt == self.config.max_retries:
                    raise failure from None
                self._sleep(self.config.retry_delay)
            else:
                break
        # Extraction and verdict validation are deliberately outside retry handling.
        result = self._extract(response)
        parse_verdict(result.text, strict=True)
        return JudgeResponse(
            result.text, result.model, {"attempts": attempt + 1, "strict_output": True}
        )

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                raise ProviderRequestError(f"{self.name}: could not close SDK client") from None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
