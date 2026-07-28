import time
from collections.abc import Iterator
from typing import Any
import os

from openai import OpenAI

from src.clients.base import BaseLLMClient
from src.errors import normalize_provider_exception
from src.models.model_response import ModelResponse
from src.models.stream_event import LLMStreamEvent


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of the shared LLM client interface."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if resolved_api_key is None or not resolved_api_key.strip():
            raise ValueError(
                "OpenAI API key was not provided and "
                "OPENAI_API_KEY is not set"
            )

        default_model = kwargs.pop("default_model", None)

        if model is not None and default_model is not None:
            raise ValueError(
                "Pass either model or default_model, not both"
            )

        selected_model = model if model is not None else default_model

        if selected_model is None or not selected_model.strip():
            raise ValueError("default_model cannot be empty")

        self._client = OpenAI(
            api_key=resolved_api_key,
            **kwargs,
        )
        self._default_model = selected_model

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamEvent]:
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        selected_model = model or self._default_model

        request_options: dict[str, object] = {
            "model": selected_model,
            "input": user_prompt,
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_options["max_output_tokens"] = max_tokens

        if system_prompt is not None:
            request_options["instructions"] = system_prompt

        start_time = time.perf_counter()
        text_parts: list[str] = []

        try:
            with self._client.responses.stream(
                **request_options,
            ) as stream:
                for event in stream:
                    if (
                        getattr(event, "type", None)
                        != "response.output_text.delta"
                    ):
                        continue

                    delta = getattr(event, "delta", "") or ""

                    if not delta:
                        continue

                    text_parts.append(delta)

                    yield LLMStreamEvent(
                        type="text_delta",
                        delta=delta,
                        snapshot="".join(text_parts),
                    )

                raw_response = stream.get_final_response()
        except Exception as exc:
            raise normalize_provider_exception(
                self.provider_name,
                exc,
            ) from exc

        latency_seconds = time.perf_counter() - start_time
        usage = getattr(raw_response, "usage", None)
        input_tokens = (
            getattr(usage, "input_tokens", 0) or 0
            if usage is not None
            else 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", 0) or 0
            if usage is not None
            else 0
        )
        total_tokens = (
            getattr(usage, "total_tokens", 0) or 0
            if usage is not None
            else input_tokens + output_tokens
        )
        content = getattr(raw_response, "output_text", "") or ""

        yield LLMStreamEvent(
            type="response",
            snapshot=content,
            response=ModelResponse(
                provider=self.provider_name,
                model=getattr(raw_response, "model", selected_model),
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency_seconds,
                finish_reason=getattr(raw_response, "status", None),
                response_id=getattr(raw_response, "id", None),
                request_id=getattr(raw_response, "_request_id", None),
                raw_response=raw_response,
            ),
        )
