from abc import ABC, abstractmethod

from src.models.model_response import ModelResponse


class BaseLLMClient(ABC):
    """Shared interface implemented by every LLM provider client."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the normalized provider name."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Generate one non-streaming model response."""
        raise NotImplementedError