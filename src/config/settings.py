# src/config/settings.py
import os

def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value or not value.strip():
        raise ValueError(
            f"Required environment variable {name!r} is missing."
        )

    return value.strip()


def load_llm_settings() -> tuple[str, str]:
    provider = get_required_env("LLM_PROVIDER")
    model = get_required_env("LLM_MODEL")

    return provider, model