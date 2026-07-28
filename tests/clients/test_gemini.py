from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from src.clients.gemini import GeminiClient


def create_fake_gemini_chunk(
    *,
    text: str,
    usage_metadata: object | None = None,
    finish_reason: str | None = None,
    model_version: str = "test-model",
    response_id: str = "response-123",
) -> SimpleNamespace:
    candidates = []

    if finish_reason is not None:
        candidates = [
            SimpleNamespace(
                finish_reason=SimpleNamespace(name=finish_reason),
            )
        ]

    return SimpleNamespace(
        text=text,
        model_version=model_version,
        usage_metadata=usage_metadata,
        candidates=candidates,
        response_id=response_id,
    )


@patch("src.clients.gemini.genai.Client")
def test_generate_streams_deltas_and_final_response(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content_stream.return_value = iter(
        [
            create_fake_gemini_chunk(text="Hello "),
            create_fake_gemini_chunk(
                text="Gemini",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=14,
                    candidates_token_count=9,
                    total_token_count=23,
                ),
                finish_reason="STOP",
            ),
        ]
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    events = list(
        client.generate(
            "Test prompt",
            system_prompt="Follow instructions",
            temperature=0.6,
            max_tokens=300,
        )
    )

    assert [event.delta for event in events[:-1]] == [
        "Hello ",
        "Gemini",
    ]

    response = events[-1].response

    assert response is not None
    assert response.provider == "gemini"
    assert response.model == "test-model"
    assert response.content == "Hello Gemini"
    assert response.input_tokens == 14
    assert response.output_tokens == 9
    assert response.total_tokens == 23
    assert response.finish_reason == "STOP"
    assert response.response_id == "response-123"

    call_kwargs = (
        mock_sdk_client.models.generate_content_stream.call_args.kwargs
    )
    config = call_kwargs["config"]

    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["contents"] == "Test prompt"
    assert config.temperature == 0.6
    assert config.max_output_tokens == 300
    assert config.system_instruction == "Follow instructions"


@patch("src.clients.gemini.genai.Client")
def test_constructor_translates_timeout_and_retries(
    mock_genai_client_class: MagicMock,
) -> None:
    GeminiClient(
        api_key="test-key",
        default_model="test-model",
        timeout=45,
        max_retries=4,
    )

    call_kwargs = mock_genai_client_class.call_args.kwargs
    http_options = call_kwargs["http_options"]

    assert call_kwargs["api_key"] == "test-key"
    assert http_options.timeout == 45000
    assert http_options.retry_options.attempts == 4


@patch("src.clients.gemini.genai.Client")
def test_constructor_merges_existing_http_options(
    mock_genai_client_class: MagicMock,
) -> None:
    client_options = types.HttpOptions(
        headers={"x-test": "1"},
        timeout=10,
    )

    GeminiClient(
        api_key="test-key",
        default_model="test-model",
        http_options=client_options,
        max_retries=3,
    )

    call_kwargs = mock_genai_client_class.call_args.kwargs
    http_options = call_kwargs["http_options"]

    assert http_options.headers == {"x-test": "1"}
    assert http_options.timeout == 10
    assert http_options.retry_options.attempts == 3


@patch("src.clients.gemini.genai.Client")
def test_generate_handles_missing_usage(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content_stream.return_value = iter(
        [create_fake_gemini_chunk(text="Hello")]
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.collect_response("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Gemini API key was not provided and "
            "GEMINI_API_KEY is not set"
        ),
    ):
        GeminiClient(default_model="test-model")


def test_rejects_empty_default_model() -> None:
    with pytest.raises(ValueError, match="default_model cannot be empty"):
        GeminiClient(
            api_key="test-key",
            default_model=" ",
        )


@patch("src.clients.gemini.genai.Client")
def test_rejects_empty_prompt(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        list(client.generate(" "))


@patch("src.clients.gemini.genai.Client")
def test_rejects_empty_system_prompt(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        list(
            client.generate(
                "Test prompt",
                system_prompt=" ",
            )
        )


@patch("src.clients.gemini.genai.Client")
def test_rejects_non_positive_max_tokens(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        list(
            client.generate(
                "Test prompt",
                max_tokens=0,
            )
        )
