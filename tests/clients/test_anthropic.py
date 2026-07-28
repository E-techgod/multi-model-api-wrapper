from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.anthropic import AnthropicClient


def create_fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        content=[
            SimpleNamespace(
                type="text",
                text="Normalized Anthropic response",
            )
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
        ),
        stop_reason="end_turn",
        id="message-123",
        _request_id="request-123",
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_returns_model_response(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    mock_sdk_client.messages.create.return_value = (
        create_fake_anthropic_response()
    )

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.provider == "anthropic"
    assert response.model == "test-model"
    assert response.content == "Normalized Anthropic response"
    assert response.input_tokens == 12
    assert response.output_tokens == 8
    assert response.total_tokens == 20
    assert response.finish_reason == "end_turn"
    assert response.response_id == "message-123"
    assert response.request_id == "request-123"

    mock_sdk_client.messages.create.assert_called_once_with(
        model="test-model",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_passes_system_prompt(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    mock_sdk_client.messages.create.return_value = (
        create_fake_anthropic_response()
    )

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        system_prompt="Follow instructions",
    )

    mock_sdk_client.messages.create.assert_called_once_with(
        model="test-model",
        max_tokens=1024,
        temperature=0.0,
        system="Follow instructions",
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_uses_model_override(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value

    fake_response = create_fake_anthropic_response()
    fake_response.model = "override-model"

    mock_sdk_client.messages.create.return_value = fake_response

    client = AnthropicClient(
        api_key="test-key",
        default_model="default-model",
    )

    response = client.generate(
        "Test prompt",
        model="override-model",
    )

    assert response.model == "override-model"

    mock_sdk_client.messages.create.assert_called_once_with(
        model="override-model",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_passes_custom_parameters(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value
    mock_sdk_client.messages.create.return_value = (
        create_fake_anthropic_response()
    )

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        temperature=0.4,
        max_tokens=250,
    )

    mock_sdk_client.messages.create.assert_called_once_with(
        model="test-model",
        max_tokens=250,
        temperature=0.4,
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_joins_multiple_text_blocks(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value

    fake_response = create_fake_anthropic_response()
    fake_response.content = [
        SimpleNamespace(type="text", text="First "),
        SimpleNamespace(type="tool_use", name="search"),
        SimpleNamespace(type="text", text="second"),
    ]

    mock_sdk_client.messages.create.return_value = fake_response

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == "First second"


@patch("src.clients.anthropic.Anthropic")
def test_generate_returns_empty_content_when_no_text_blocks(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value

    fake_response = create_fake_anthropic_response()
    fake_response.content = [
        SimpleNamespace(type="tool_use", name="search"),
    ]

    mock_sdk_client.messages.create.return_value = fake_response

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Anthropic API key was not provided and "
            "ANTHROPIC_API_KEY is not set"
        ),
    ):
        AnthropicClient(
            default_model="test-model",
        )


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
        client.generate(" ")


@patch("src.clients.anthropic.Anthropic")
def test_rejects_empty_system_prompt(
    mock_anthropic_class: MagicMock,
) -> None:
    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        client.generate(
            "Test prompt",
            system_prompt=" ",
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
        client.generate(
            "Test prompt",
            max_tokens=0,
        )


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "env-test-key"},
    clear=True,
)
@patch("src.clients.anthropic.Anthropic")
def test_uses_api_key_from_environment(
    mock_anthropic_class: MagicMock,
) -> None:
    AnthropicClient(default_model="test-model")

    mock_anthropic_class.assert_called_once_with(
        api_key="env-test-key",
    )


@patch("src.clients.anthropic.Anthropic")
def test_generate_handles_missing_usage(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value

    fake_response = create_fake_anthropic_response()
    fake_response.usage = None

    mock_sdk_client.messages.create.return_value = fake_response

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.anthropic.Anthropic")
def test_generate_handles_missing_content(
    mock_anthropic_class: MagicMock,
) -> None:
    mock_sdk_client = mock_anthropic_class.return_value

    fake_response = create_fake_anthropic_response()
    del fake_response.content

    mock_sdk_client.messages.create.return_value = fake_response

    client = AnthropicClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""
