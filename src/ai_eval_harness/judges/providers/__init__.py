"""Optional providers. Importing this module does not import either SDK."""

from typing import Any

from ai_eval_harness.judges.providers.anthropic import AnthropicJudgeProvider
from ai_eval_harness.judges.providers.common import LiveJudgeProvider
from ai_eval_harness.judges.providers.config import ProviderConfig
from ai_eval_harness.judges.providers.openai import OpenAIJudgeProvider


def build_provider(
    config: ProviderConfig, *, client: Any = None, api_key: str | None = None
) -> LiveJudgeProvider:
    provider = OpenAIJudgeProvider if config.provider == "openai" else AnthropicJudgeProvider
    return provider(config, client=client, api_key=api_key)


__all__ = ["AnthropicJudgeProvider", "OpenAIJudgeProvider", "ProviderConfig", "build_provider"]
