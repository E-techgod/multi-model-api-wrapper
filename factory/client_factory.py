# factory/client_factory.py

from typing import Any

from src.clients.base import BaseLLMClient
from src.clients.openai import OpenAIClient
from src.clients.anthropic import AnthropicClient
from src.clients.gemini import GeminiClient
from src.clients.groq import GroqClient

from factory.providers import LLMProvider


class ClientFactory:
    """Creates provider-specific LLM clients behind a shared interface."""

    @staticmethod
    def create(
        provider: str | LLMProvider,
        model: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMClient:
        """
        Create an LLM client for the requested provider.

        Args:
            provider:
                Provider name or LLMProvider enum value.
            model:
                Model identifier used by the provider.
            api_key:
                Optional API key. Individual clients may load it from the
                environment when this value is not provided.
            **kwargs:
                Additional provider-specific configuration.

        Returns:
            A concrete implementation of BaseLLMClient.

        Raises:
            ValueError:
                If the provider is empty or unsupported.
        """
        normalized_provider = ClientFactory._normalize_provider(provider)

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

        # Defensive fallback. _normalize_provider should reject unsupported values.
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}"
        )

    @staticmethod
    def _normalize_provider(
        provider: str | LLMProvider,
    ) -> LLMProvider:
        """Normalize string input into an LLMProvider enum."""

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
                provider.value for provider in LLMProvider
            )

            raise ValueError(
                f"Unsupported LLM provider: {provider!r}. "
                f"Supported providers: {supported}."
            ) from exc