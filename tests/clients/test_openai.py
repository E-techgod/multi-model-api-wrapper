from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.openai import OpenAIClient


def create_fake_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        output_text="Normalized test response",
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
def test_generate_returns_model_response(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    mock_sdk_client.responses.create.return_value = (
        create_fake_openai_response()
    )

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.provider == "openai"
    assert response.model == "test-model"
    assert response.content == "Normalized test response"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "completed"
    assert response.response_id == "response-123"
    assert response.request_id == "request-123"

    mock_sdk_client.responses.create.assert_called_once_with(
        model="test-model",
        input="Test prompt",
    )


@patch("src.clients.openai.OpenAI")
def test_generate_uses_model_override(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value

    fake_response = create_fake_openai_response()
    fake_response.model = "override-model"

    mock_sdk_client.responses.create.return_value = fake_response

    client = OpenAIClient(
        api_key="test-key",
        default_model="default-model",
    )

    response = client.generate(
        "Test prompt",
        model="override-model",
    )

    assert response.model == "override-model"

    mock_sdk_client.responses.create.assert_called_once_with(
        model="override-model",
        input="Test prompt",
    )


@patch("src.clients.openai.OpenAI")
def test_generate_passes_max_tokens(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value
    mock_sdk_client.responses.create.return_value = (
        create_fake_openai_response()
    )

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        max_tokens=100,
    )

    mock_sdk_client.responses.create.assert_called_once_with(
        model="test-model",
        input="Test prompt",
        max_output_tokens=100,
    )


@patch("src.clients.openai.OpenAI")
def test_generate_handles_missing_usage(
    mock_openai_class: MagicMock,
) -> None:
    mock_sdk_client = mock_openai_class.return_value

    fake_response = create_fake_openai_response()
    fake_response.usage = None

    mock_sdk_client.responses.create.return_value = fake_response

    client = OpenAIClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        OpenAIClient(
            api_key=" ",
            default_model="test-model",
        )


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

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        client.generate(" ")