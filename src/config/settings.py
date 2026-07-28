# src/config/settings.py
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 2

    def client_options(self) -> dict[str, Any]:
        return {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value or not value.strip():
        raise ValueError(f"Required environment variable {name!r} is missing.")

    return value.strip()


def load_llm_settings() -> LLMSettings:
    return LLMSettings(
        provider=get_required_env("LLM_PROVIDER"),
        model=get_required_env("LLM_MODEL"),
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
