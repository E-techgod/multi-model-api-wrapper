# src/factory/client_factory.py

from typing import Any

from src.clients.base import BaseLLMClient
from src.clients.openai import OpenAIClient
from src.clients.anthropic import AnthropicClient
from src.clients.gemini import GeminiClient
from src.clients.groq import GroqClient
from src.factory.providers import LLMProvider


class ClientFactory:
    """Create provider-specific LLM clients behind a shared interface."""

    @staticmethod
    def create(provider: str | LLMProvider,model: str,api_key: str | None = None,**kwargs: Any) -> BaseLLMClient:
        """
        Create an LLM client for the requested provider.

        Args:
            provider:
                Provider name or LLMProvider enum member.
            model:
                Provider-specific model identifier.
            api_key:
                Optional API key. The concrete client may load it
                from an environment variable when omitted.
            **kwargs:
                Additional configuration passed to the concrete client.

        Returns:
            A concrete BaseLLMClient implementation.

        Raises:
            ValueError:
                If the provider is empty, invalid, or unsupported.
        """
        normalized_provider = ClientFactory._normalize_provider(
            provider
        )

        if normalized_provider == LLMProvider.OPENAI:
            return OpenAIClient(
                model=model,
                api_key=api_key,
                **kwargs,
            )

        if normalized_provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                model=model,
                api_key=api_key,
                **kwargs,
            )

        if normalized_provider == LLMProvider.GEMINI:
            return GeminiClient(
                model=model,
                api_key=api_key,
                **kwargs,
            )

        if normalized_provider == LLMProvider.GROQ:
            return GroqClient(
                model=model,
                api_key=api_key,
                **kwargs,
            )

        raise ValueError(
            f"Unsupported LLM provider: {provider!r}"
        )

    @staticmethod
    def _normalize_provider(provider: str | LLMProvider) -> LLMProvider:
        """Normalize a provider string into an LLMProvider member."""

        if isinstance(provider, LLMProvider):
            return provider

        if not isinstance(provider, str):
            raise ValueError(
                "Provider must be a string or LLMProvider value."
            )

        normalized = provider.strip().lower()

        if not normalized:
            raise ValueError("Provider cannot be empty.")

        try:
            return LLMProvider(normalized)
        except ValueError as exc:
            supported = ", ".join(
                member.value for member in LLMProvider
            )

            raise ValueError(
                f"Unsupported LLM provider: {provider!r}. "
                f"Supported providers: {supported}."
            ) from exc

    @staticmethod
    def supported_providers() -> tuple[str, ...]:
        """Return all supported provider names."""

        return tuple(
            provider.value for provider in LLMProvider
        )