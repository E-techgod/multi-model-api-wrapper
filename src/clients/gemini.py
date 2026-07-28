import time
from typing import Any
import os

from google import genai
from google.genai import types

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse


class GeminiClient(BaseLLMClient):
    """Gemini implementation of the shared LLM client interface."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if resolved_api_key is None or not resolved_api_key.strip():
            raise ValueError(
                "Gemini API key was not provided and "
                "GEMINI_API_KEY is not set"
            )

        default_model = kwargs.pop("default_model", None)

        if model is not None and default_model is not None:
            raise ValueError(
                "Pass either model or default_model, not both"
            )

        selected_model = model if model is not None else default_model

        if selected_model is None or not selected_model.strip():
            raise ValueError("default_model cannot be empty")

        self._client = genai.Client(
            api_key=resolved_api_key,
            **kwargs,
        )
        self._default_model = selected_model

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        selected_model = model or self._default_model

        config_options: dict[str, Any] = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            config_options["max_output_tokens"] = max_tokens

        if system_prompt is not None:
            config_options["system_instruction"] = system_prompt

        config = types.GenerateContentConfig(**config_options)

        start_time = time.perf_counter()

        raw_response = self._client.models.generate_content(
            model=selected_model,
            contents=user_prompt,
            config=config,
        )

        latency_seconds = time.perf_counter() - start_time

        usage = getattr(raw_response, "usage_metadata", None)

        input_tokens = (
            getattr(usage, "prompt_token_count", 0) or 0
            if usage is not None
            else 0
        )
        output_tokens = (
            getattr(usage, "candidates_token_count", 0) or 0
            if usage is not None
            else 0
        )
        total_tokens = (
            getattr(usage, "total_token_count", 0) or 0
            if usage is not None
            else 0
        )

        return ModelResponse(
            provider=self.provider_name,
            model=getattr(raw_response, "model_version", selected_model),
            content=self._extract_text(raw_response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            finish_reason=self._extract_finish_reason(raw_response),
            response_id=getattr(raw_response, "response_id", None),
            request_id=None,
            raw_response=raw_response,
        )

    @staticmethod
    def _extract_text(raw_response: Any) -> str:
        """Extract generated text without assuming response.text always exists."""

        try:
            return raw_response.text or ""
        except (AttributeError, ValueError):
            return ""

    @staticmethod
    def _extract_finish_reason(raw_response: Any) -> str | None:
        """Extract the first candidate's finish reason safely."""

        candidates = getattr(raw_response, "candidates", None)

        if not candidates:
            return None

        finish_reason = getattr(candidates[0], "finish_reason", None)

        if finish_reason is None:
            return None

        return getattr(finish_reason, "name", str(finish_reason))
