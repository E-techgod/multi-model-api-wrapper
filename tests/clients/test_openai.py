from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.openai import OpenAIClient
from src.errors import RateLimitError


class FakeOpenAIStream:
    def __init__(
        self,
        events: list[SimpleNamespace],
        final_response: SimpleNamespace,
    ) -> None:
        self._events = events
        self._final_response = final_response

    def __enter__(self) -> "FakeOpenAIStream":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        return None

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self) -> SimpleNamespace:
        return self._final_response


def create_fake_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        output_text="Hello world",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
        status="completed",
        id="response-123",
        _request_id="request-123",
    )


@patch("src.clients.openai.OpenAI")
def test_generate_streams_deltas_and_final_response(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    final_response = create_fake_openai_response()
    mock_sdk_client.responses.stream.return_value = FakeOpenAIStream(
        events=[
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta=" world",
            ),
        ],
        final_response=final_response,
    )

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    events = list(
        client.generate(
            "Test prompt",
            system_prompt="Follow instructions",
            temperature=0.4,
            max_tokens=100,
        )
    )

    assert [event.delta for event in events[:-1]] == [
        "Hello",
        " world",
    ]

    response = events[-1].response

    assert response is not None
    assert response.provider == "openai"
    assert response.model == "test-model"
    assert response.content == "Hello world"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "completed"
    assert response.response_id == "response-123"
    assert response.request_id == "request-123"

    mock_sdk_client.responses.stream.assert_called_once_with(
        model="test-model",
        input="Test prompt",
        temperature=0.4,
        max_output_tokens=100,
        instructions="Follow instructions",
    )


@patch("src.clients.openai.OpenAI")
def test_collect_response_returns_final_response(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    final_response = create_fake_openai_response()
    mock_sdk_client.responses.stream.return_value = FakeOpenAIStream(
        events=[],
        final_response=final_response,
    )

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.collect_response("Test prompt")

    assert response.content == "Hello world"


@patch("src.clients.openai.OpenAI")
def test_generate_handles_missing_usage(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    final_response = create_fake_openai_response()
    final_response.usage = None
    mock_sdk_client.responses.stream.return_value = FakeOpenAIStream(
        events=[],
        final_response=final_response,
    )

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.collect_response("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.openai.OpenAI")
def test_generate_normalizes_provider_rate_limit_errors(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    mock_sdk_client.responses.stream.side_effect = type(
        "RateLimitError",
        (Exception,),
        {},
    )("too many requests")

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(RateLimitError, match="too many requests"):
        list(client.generate("Test prompt"))


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "OpenAI API key was not provided and "
            "OPENAI_API_KEY is not set"
        ),
    ):
        OpenAIClient(default_model="test-model")


def test_rejects_empty_default_model() -> None:
    with pytest.raises(ValueError, match="default_model cannot be empty"):
        OpenAIClient(
            api_key="test-key",
            default_model=" ",
        )


@patch("src.clients.openai.OpenAI")
def test_rejects_empty_prompt(
    mock_openai_class: MagicMock,
) -> None:
    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        list(client.generate(" "))


@patch("src.clients.openai.OpenAI")
def test_rejects_empty_system_prompt(
    mock_openai_class: MagicMock,
) -> None:
    client = OpenAIClient(
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


@patch("src.clients.openai.OpenAI")
def test_rejects_non_positive_max_tokens(
    mock_openai_class: MagicMock,
) -> None:
    client = OpenAIClient(
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
