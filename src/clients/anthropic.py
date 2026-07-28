import time
from typing import Any

from anthropic import Anthropic

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse


class AnthropicClient(BaseLLMClient):
    """Anthropic implementation of the shared LLM client interface."""

    def __init__(self,api_key: str,default_model: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key cannot be empty")

        if not default_model.strip():
            raise ValueError("default_model cannot be empty")

        self._client = Anthropic(api_key=api_key)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(self,prompt: str,*, model: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> ModelResponse:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        selected_model = model or self._default_model
        selected_max_tokens = max_tokens or 1024

        start_time = time.perf_counter()

        raw_response = self._client.messages.create(
            model=selected_model,
            max_tokens=selected_max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        latency_seconds = time.perf_counter() - start_time

        input_tokens = raw_response.usage.input_tokens
        output_tokens = raw_response.usage.output_tokens

        return ModelResponse(
            provider=self.provider_name,
            model=raw_response.model,
            content=self._extract_text(raw_response.content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=latency_seconds,
            finish_reason=raw_response.stop_reason,
            response_id=raw_response.id,
            request_id=getattr(raw_response, "_request_id", None),
            raw_response=raw_response,
        )

    @staticmethod
    def _extract_text(content_blocks: list[Any]) -> str:
        """Join all textual content blocks returned by Anthropic."""

        text_parts: list[str] = []

        for block in content_blocks:
            if getattr(block, "type", None) != "text":
                continue

            text = getattr(block, "text", None)

            if text:
                text_parts.append(text)

        return "".join(text_parts)