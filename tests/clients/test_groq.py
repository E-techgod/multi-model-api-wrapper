from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.groq import GroqClient
from src.errors import ProviderUnavailableError


def create_fake_groq_chunk(
    *,
    text: str | None,
    finish_reason: str | None = None,
    usage: object | None = None,
    model: str = "test-model",
    response_id: str = "completion-123",
    request_id: str = "request-123",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        model=model,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        _request_id=request_id,
    )


@patch("src.clients.groq.Groq")
def test_generate_streams_deltas_and_final_response(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value
    mock_sdk_client.chat.completions.create.return_value = iter(
        [
            create_fake_groq_chunk(text="Hello "),
            create_fake_groq_chunk(
                text="Groq",
                finish_reason="stop",
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                ),
            ),
        ]
    )

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    events = list(
        client.generate(
            "Test prompt",
            system_prompt="Follow instructions",
            temperature=0.5,
            max_tokens=300,
        )
    )

    assert [event.delta for event in events[:-1]] == [
        "Hello ",
        "Groq",
    ]

    response = events[-1].response

    assert response is not None
    assert response.provider == "groq"
    assert response.model == "test-model"
    assert response.content == "Hello Groq"
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
                "role": "system",
                "content": "Follow instructions",
            },
            {
                "role": "user",
                "content": "Test prompt",
            },
        ],
        temperature=0.5,
        max_tokens=300,
        stream=True,
    )


@patch("src.clients.groq.Groq")
def test_generate_handles_missing_usage(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value
    mock_sdk_client.chat.completions.create.return_value = iter(
        [create_fake_groq_chunk(text="Hello")]
    )

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.collect_response("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.groq.Groq")
def test_generate_normalizes_provider_availability_errors(
    mock_groq_class: MagicMock,
) -> None:
    mock_sdk_client = mock_groq_class.return_value
    mock_sdk_client.chat.completions.create.side_effect = type(
        "APIConnectionError",
        (Exception,),
        {},
    )("connection failed")

    client = GroqClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(
        ProviderUnavailableError,
        match="connection failed",
    ):
        list(client.generate("Test prompt"))


def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key cannot be empty"):
        GroqClient(
            api_key=" ",
            default_model="test-model",
        )


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=("Groq API key was not provided and " "GROQ_API_KEY is not set"),
    ):
        GroqClient(default_model="test-model")


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

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        list(client.generate(" "))


@patch("src.clients.groq.Groq")
def test_rejects_empty_system_prompt(
    mock_groq_class: MagicMock,
) -> None:
    client = GroqClient(
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
        list(
            client.generate(
                "Test prompt",
                max_tokens=0,
            )
        )
