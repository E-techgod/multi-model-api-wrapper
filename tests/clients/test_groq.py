from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.groq import GroqClient


def create_fake_groq_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="completion-123",
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Normalized Groq response",
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
        _request_id="request-123",
    )


@patch("src.clients.groq.Groq")
def test_generate_returns_model_response(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value
    mock_sdk_client.chat.completions.create.return_value = (
        create_fake_groq_response()
    )

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.provider == "groq"
    assert response.model == "test-model"
    assert response.content == "Normalized Groq response"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.total_tokens == 18
    assert response.finish_reason == "stop"
    assert response.response_id == "completion-123"
    assert response.request_id == "request-123"

    mock_sdk_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
        temperature=0.0,
    )


@patch("src.clients.groq.Groq")
def test_generate_uses_model_override(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value

    fake_response = create_fake_groq_response()
    fake_response.model = "override-model"

    mock_sdk_client.chat.completions.create.return_value = fake_response

    client = GroqClient(
        api_key="test-key",
        default_model="default-model",
    )

    response = client.generate(
        "Test prompt",
        model="override-model",
    )

    assert response.model == "override-model"

    mock_sdk_client.chat.completions.create.assert_called_once_with(
        model="override-model",
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
        temperature=0.0,
    )


@patch("src.clients.groq.Groq")
def test_generate_passes_custom_parameters(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value
    mock_sdk_client.chat.completions.create.return_value = (
        create_fake_groq_response()
    )

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        temperature=0.5,
        max_tokens=300,
    )

    mock_sdk_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
        temperature=0.5,
        max_tokens=300,
    )


@patch("src.clients.groq.Groq")
def test_generate_handles_missing_usage(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value

    fake_response = create_fake_groq_response()
    fake_response.usage = None

    mock_sdk_client.chat.completions.create.return_value = fake_response

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.groq.Groq")
def test_generate_handles_empty_choices(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value

    fake_response = create_fake_groq_response()
    fake_response.choices = []

    mock_sdk_client.chat.completions.create.return_value = fake_response

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""
    assert response.finish_reason is None


@patch("src.clients.groq.Groq")
def test_generate_handles_missing_message(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value

    fake_response = create_fake_groq_response()
    fake_response.choices = [
        SimpleNamespace(
            message=None,
            finish_reason="stop",
        )
    ]

    mock_sdk_client.chat.completions.create.return_value = fake_response

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""
    assert response.finish_reason == "stop"


@patch("src.clients.groq.Groq")
def test_generate_handles_none_content(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value

    fake_response = create_fake_groq_response()
    fake_response.choices[0].message.content = None

    mock_sdk_client.chat.completions.create.return_value = fake_response

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""


def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        GroqClient(
            api_key=" ",
            default_model="test-model",
        )


def test_rejects_empty_default_model() -> None:
    with pytest.raises(ValueError, match="default_model cannot be empty"):
        GroqClient(
            api_key="test-key",
            default_model=" ",
        )


@patch("src.clients.groq.Groq")
def test_rejects_empty_prompt(
    mock_groq_class: MagicMock,
) -> None:
    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        client.generate(" ")


@patch("src.clients.groq.Groq")
def test_rejects_non_positive_max_tokens(
    mock_groq_class: MagicMock,
) -> None:
    client = GroqClient(
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