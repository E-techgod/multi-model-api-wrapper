from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.models.model_response import ModelResponse
from src.models.stream_event import LLMStreamEvent


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
    ) -> Iterator[LLMStreamEvent]:
        """Stream model output and finish with a normalized response."""
        raise NotImplementedError

    def collect_response(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Consume a stream and return the final normalized response."""

        final_response: ModelResponse | None = None

        for event in self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if event.response is not None:
                final_response = event.response

        if final_response is None:
            raise RuntimeError("LLM stream completed without a response")

        return final_response
