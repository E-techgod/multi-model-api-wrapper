import time
from collections.abc import Iterator
from typing import Any
import os

from google import genai
from google.genai import types

from src.clients.base import BaseLLMClient
from src.errors import normalize_provider_exception
from src.models.model_response import ModelResponse
from src.models.stream_event import LLMStreamEvent


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

        http_options = self._build_http_options(kwargs)

        client_options: dict[str, Any] = {
            "api_key": resolved_api_key,
            **kwargs,
        }

        if http_options is not None:
            client_options["http_options"] = http_options

        self._client = genai.Client(**client_options)
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
    ) -> Iterator[LLMStreamEvent]:
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
        text_parts: list[str] = []
        raw_response: Any = None

        try:
            raw_stream = self._client.models.generate_content_stream(
                model=selected_model,
                contents=user_prompt,
                config=config,
            )

            for chunk in raw_stream:
                raw_response = chunk
                delta = self._extract_text(chunk)

                if not delta:
                    continue

                text_parts.append(delta)

                yield LLMStreamEvent(
                    type="text_delta",
                    delta=delta,
                    snapshot="".join(text_parts),
                )
        except Exception as exc:
            raise normalize_provider_exception(
                self.provider_name,
                exc,
            ) from exc

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

        content = "".join(text_parts)

        yield LLMStreamEvent(
            type="response",
            snapshot=content,
            response=ModelResponse(
                provider=self.provider_name,
                model=getattr(raw_response, "model_version", selected_model),
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency_seconds,
                finish_reason=self._extract_finish_reason(raw_response),
                response_id=getattr(raw_response, "response_id", None),
                request_id=None,
                raw_response=raw_response,
            ),
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

    @staticmethod
    def _build_http_options(
        kwargs: dict[str, Any],
    ) -> types.HttpOptions | dict[str, Any] | None:
        """Translate shared client options into Gemini HTTP options."""

        timeout = kwargs.pop("timeout", None)
        max_retries = kwargs.pop("max_retries", None)
        http_options = kwargs.pop("http_options", None)

        if (
            timeout is None
            and max_retries is None
            and http_options is None
        ):
            return None

        if isinstance(http_options, types.HttpOptions):
            http_options_data = http_options.model_dump(
                exclude_none=True
            )
        elif isinstance(http_options, dict):
            http_options_data = dict(http_options)
        elif http_options is None:
            http_options_data = {}
        else:
            raise ValueError(
                "http_options must be a Gemini HttpOptions "
                "instance or dict"
            )

        retry_options = http_options_data.get("retry_options")

        if max_retries is not None:
            if max_retries < 0:
                raise ValueError("max_retries cannot be negative")
            if retry_options is None:
                retry_options = {}
            elif isinstance(retry_options, types.HttpRetryOptions):
                retry_options = retry_options.model_dump(
                    exclude_none=True
                )
            else:
                retry_options = dict(retry_options)

            retry_options["attempts"] = max_retries
            http_options_data["retry_options"] = retry_options

        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout must be greater than zero")

            # google.genai HttpOptions.timeout is in milliseconds, while the
            # rest of this wrapper (and the other provider SDKs) use seconds.
            http_options_data["timeout"] = timeout * 1000

        return types.HttpOptions(**http_options_data)
