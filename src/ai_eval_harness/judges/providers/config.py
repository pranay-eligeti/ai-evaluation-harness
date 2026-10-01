"""Safe provider configuration: no credential fields and no arbitrary endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1)
    timeout: float = Field(default=30.0, gt=0, le=300, allow_inf_nan=False)
    max_retries: int = Field(default=1, ge=0, le=3)
    retry_delay: float = Field(default=1.0, ge=0, le=10, allow_inf_nan=False)
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)
    prompt_version: Literal["2"] = "2"

    @field_validator("model")
    @classmethod
    def nonblank_model(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Model must be nonblank and have no surrounding whitespace")
        return value
