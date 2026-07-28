from typing import Any

from src.clients.base import BaseLLMClient
from src.factory.client_factory import ClientFactory


def build_llm_client(provider: str,model: str,api_key: str | None = None,*kwargs: Any) -> BaseLLMClient:
    """Build an LLM client using provider-independent configuration."""

    return ClientFactory.create(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs,
    )