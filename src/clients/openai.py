import time
from typing import Any

from openai import OpenAI

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of the shared LLM client interface."""

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

        self._client = OpenAI(api_key=api_key, **kwargs)
        self._default_model = selected_model

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,prompt: str, *, model: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> ModelResponse:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        selected_model = model or self._default_model

        request_options: dict[str, object] = {
            "model": selected_model,
            "input": prompt,
        }

        if max_tokens is not None:
            request_options["max_output_tokens"] = max_tokens

        start_time = time.perf_counter()

        raw_response = self._client.responses.create(
            **request_options,
        )

        latency_seconds = time.perf_counter() - start_time

        usage = raw_response.usage

        input_tokens = usage.input_tokens if usage is not None else 0
        output_tokens = usage.output_tokens if usage is not None else 0
        total_tokens = usage.total_tokens if usage is not None else 0

        return ModelResponse(
            provider=self.provider_name,
            model=raw_response.model,
            content=raw_response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            finish_reason=raw_response.status,
            response_id=raw_response.id,
            request_id=getattr(raw_response, "_request_id", None),
            raw_response=raw_response,
        )
