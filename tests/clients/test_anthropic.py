from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.anthropic import AnthropicClient
from src.errors import TimeoutError


class FakeAnthropicStream:
    def __init__(
        self,
        text_stream: list[str],
        final_message: SimpleNamespace,
        request_id: str | None = "request-123",
    ) -> None:
        self.text_stream = iter(text_stream)
        self._final_message = final_message
        self.request_id = request_id

    def __enter__(self) -> "FakeAnthropicStream":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self._final_message


def create_fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        content=[
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="Claude"),
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
        ),
        stop_reason="end_turn",
        id="message-123",
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_streams_deltas_and_final_response(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    final_message = create_fake_anthropic_response()
    mock_sdk_client.messages.stream.return_value = FakeAnthropicStream(
        text_stream=["Hello ", "Claude"],
        final_message=final_message,
    )

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    events = list(
        client.generate(
            "Test prompt",
            system_prompt="Follow instructions",
            temperature=0.4,
            max_tokens=250,
        )
    )

    assert [event.delta for event in events[:-1]] == [
        "Hello ",
        "Claude",
    ]

    response = events[-1].response

    assert response is not None
    assert response.provider == "anthropic"
    assert response.model == "test-model"
    assert response.content == "Hello Claude"
    assert response.input_tokens == 12
    assert response.output_tokens == 8
    assert response.total_tokens == 20
    assert response.finish_reason == "end_turn"
    assert response.response_id == "message-123"
    assert response.request_id == "request-123"

    mock_sdk_client.messages.stream.assert_called_once_with(
        model="test-model",
        max_tokens=250,
        temperature=0.4,
        system="Follow instructions",
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_handles_missing_usage(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    final_message = create_fake_anthropic_response()
    final_message.usage = None
    mock_sdk_client.messages.stream.return_value = FakeAnthropicStream(
        text_stream=[],
        final_message=final_message,
    )

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.collect_response("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.anthropic.Anthropic")
def test_generate_normalizes_provider_timeout_errors(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    mock_sdk_client.messages.stream.side_effect = type(
        "APITimeoutError",
        (Exception,),
        {},
    )("request timed out")

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(TimeoutError, match="request timed out"):
        list(client.generate("Test prompt"))


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Anthropic API key was not provided and "
            "ANTHROPIC_API_KEY is not set"
        ),
    ):
        AnthropicClient(default_model="test-model")


def test_rejects_empty_default_model() -> None:
    with pytest.raises(ValueError, match="default_model cannot be empty"):
        AnthropicClient(
            api_key="test-key",
            default_model=" ",
        )


@patch("src.clients.anthropic.Anthropic")
def test_rejects_empty_prompt(
    mock_anthropic_class: MagicMock,
) -> None:
    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        list(client.generate(" "))


@patch("src.clients.anthropic.Anthropic")
def test_rejects_empty_system_prompt(
    mock_anthropic_class: MagicMock,
) -> None:
    client = AnthropicClient(
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


@patch("src.clients.anthropic.Anthropic")
def test_rejects_non_positive_max_tokens(
    mock_anthropic_class: MagicMock,
) -> None:
    client = AnthropicClient(
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
