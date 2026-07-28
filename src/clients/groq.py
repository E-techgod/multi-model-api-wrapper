import time
from typing import Any

from groq import Groq

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse


class GroqClient(BaseLLMClient):
    """Groq implementation of the shared LLM client interface."""

    def __init__(self,model: str | None = None,api_key: str | None = None,**kwargs: Any) -> None:
        if api_key is None or not api_key.strip():
            raise ValueError("api_key cannot be empty")

        default_model = kwargs.pop("default_model", None)

        if model is not None and default_model is not None:
            raise ValueError(
                "Pass either model or default_model, not both"
            )

        selected_model = model if model is not None else default_model

        if selected_model is None or not selected_model.strip():
            raise ValueError("default_model cannot be empty")

        self._client = Groq(api_key=api_key, **kwargs)
        self._default_model = selected_model

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        selected_model = model or self._default_model

        request_options: dict[str, object] = {
            "model": selected_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_options["max_tokens"] = max_tokens

        start_time = time.perf_counter()

        raw_response = self._client.chat.completions.create(
            **request_options,
        )

        latency_seconds = time.perf_counter() - start_time

        choice = self._get_first_choice(raw_response)
        usage = getattr(raw_response, "usage", None)

        input_tokens = (
            getattr(usage, "prompt_tokens", 0) or 0
            if usage is not None
            else 0
        )
        output_tokens = (
            getattr(usage, "completion_tokens", 0) or 0
            if usage is not None
            else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0) or 0
            if usage is not None
            else 0
        )

        return ModelResponse(
            provider=self.provider_name,
            model=getattr(raw_response, "model", selected_model),
            content=self._extract_content(choice),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            finish_reason=getattr(choice, "finish_reason", None),
            response_id=getattr(raw_response, "id", None),
            request_id=getattr(raw_response, "_request_id", None),
            raw_response=raw_response,
        )

    @staticmethod
    def _get_first_choice(raw_response: object) -> object | None:
        """Return the first completion choice when available."""

        choices = getattr(raw_response, "choices", None)

        if not choices:
            return None

        return choices[0]

    @staticmethod
    def _extract_content(choice: object | None) -> str:
        """Extract text from a Groq completion choice safely."""

        if choice is None:
            return ""

        message = getattr(choice, "message", None)

        if message is None:
            return ""

        content = getattr(message, "content", None)

        return content or ""
