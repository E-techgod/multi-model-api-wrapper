# tests/test_client_factory.py
import pytest
from unittest.mock import patch

from src.clients.anthropic import AnthropicClient
from src.clients.gemini import GeminiClient
from src.clients.groq import GroqClient
from src.clients.openai import OpenAIClient
from src.factory.client_factory import ClientFactory
from src.factory.providers import LLMProvider
from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings

def test_create_openai_client() -> None:
    client = ClientFactory.create(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
    )

    assert isinstance(client, OpenAIClient)

def test_create_anthropic_client() -> None:
    client = ClientFactory.create(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
    )

    assert isinstance(client, AnthropicClient)


def test_create_gemini_client() -> None:
    client = ClientFactory.create(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="test-key",
    )

    assert isinstance(client, GeminiClient)

def test_create_groq_client() -> None:
    client = ClientFactory.create(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key="test-key",
    )

    assert isinstance(client, GroqClient)

def test_create_client_accepts_provider_enum() -> None:
    client = ClientFactory.create(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1-mini",
        api_key="test-key",
    )

    assert isinstance(client, OpenAIClient)

def test_create_client_normalizes_provider_name() -> None:
    client = ClientFactory.create(
        provider="  OPENAI  ",
        model="gpt-4.1-mini",
        api_key="test-key",
    )

    assert isinstance(client, OpenAIClient)

def test_create_client_rejects_unsupported_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported LLM provider",
    ):
        ClientFactory.create(
            provider="cohere",
            model="some-model",
            api_key="test-key",
        )

def test_create_client_rejects_empty_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Provider cannot be empty",
    ):
        ClientFactory.create(
            provider="   ",
            model="some-model",
            api_key="test-key",
        )

def test_supported_providers() -> None:
    assert ClientFactory.supported_providers() == (
        "openai",
        "anthropic",
        "gemini",
        "groq",
    )


@patch("src.factory.client_factory.OpenAIClient")
def test_factory_selects_openai_client(
    mock_openai_client,
) -> None:
    ClientFactory.create(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-key",
    )

    mock_openai_client.assert_called_once_with(
        model="gpt-4.1-mini",
        api_key="test-key",
    )

@patch("src.factory.client_factory.OpenAIClient")
def test_create_openai_client(mock_openai_client) -> None:
    mock_instance = mock_openai_client.return_value

    result = ClientFactory.create(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
    )

    mock_openai_client.assert_called_once_with(
        model="gpt-test",
        api_key="test-key",
    )
    assert result is mock_instance

@patch("src.factory.client_factory.AnthropicClient")
def test_create_anthropic_client(mock_anthropic_client) -> None:
    mock_instance = mock_anthropic_client.return_value

    result = ClientFactory.create(
        provider="anthropic",
        model="claude-test",
        api_key="test-key",
    )

    mock_anthropic_client.assert_called_once_with(
        model="claude-test",
        api_key="test-key",
    )
    assert result is mock_instance

@patch("src.factory.client_factory.GeminiClient")
def test_create_gemini_client(mock_gemini_client) -> None:
    mock_instance = mock_gemini_client.return_value

    result = ClientFactory.create(
        provider="gemini",
        model="gemini-test",
        api_key="test-key",
    )

    mock_gemini_client.assert_called_once_with(
        model="gemini-test",
        api_key="test-key",
    )
    assert result is mock_instance

@patch("src.factory.client_factory.GroqClient")
def test_create_groq_client(mock_groq_client) -> None:
    mock_instance = mock_groq_client.return_value

    result = ClientFactory.create(
        provider="groq",
        model="llama-test",
        api_key="test-key",
    )

    mock_groq_client.assert_called_once_with(
        model="llama-test",
        api_key="test-key",
    )
    assert result is mock_instance


@patch("src.factory.client_factory.OpenAIClient")
def test_create_forwards_optional_arguments(
    mock_openai_client,
) -> None:
    ClientFactory.create(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=3,
    )

    mock_openai_client.assert_called_once_with(
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=3,
    )

@patch("src.factory.client_factory.OpenAIClient")
def test_create_normalizes_provider_string(
    mock_openai_client,
) -> None:
    ClientFactory.create(
        provider="  OPENAI  ",
        model="gpt-test",
        api_key="test-key",
    )

    mock_openai_client.assert_called_once_with(
        model="gpt-test",
        api_key="test-key",
    )

@patch("src.factory.client_factory.AnthropicClient")
def test_create_accepts_provider_enum(
    mock_anthropic_client,
) -> None:
    ClientFactory.create(
        provider=LLMProvider.ANTHROPIC,
        model="claude-test",
        api_key="test-key",
    )

    mock_anthropic_client.assert_called_once_with(
        model="claude-test",
        api_key="test-key",
    )

def test_create_rejects_empty_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Provider cannot be empty",
    ):
        ClientFactory.create(
            provider="   ",
            model="test-model",
            api_key="test-key",
        )

def test_create_rejects_unsupported_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported LLM provider",
    ):
        ClientFactory.create(
            provider="cohere",
            model="test-model",
            api_key="test-key",
        )

def test_create_rejects_non_string_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Provider must be a string",
    ):
        ClientFactory.create(
            provider=123,  # type: ignore[arg-type]
            model="test-model",
            api_key="test-key",
        )

def test_supported_providers() -> None:
    assert ClientFactory.supported_providers() == (
        "openai",
        "anthropic",
        "gemini",
        "groq",
    )

@patch("src.config.client_builder.ClientFactory.create")
def test_build_llm_client_uses_settings(
    mock_create,
) -> None:
    settings = LLMSettings(
        provider="groq",
        model="llama-test",
        api_key="test-key",
        timeout=30.0,
        max_retries=2,
    )

    result = build_llm_client(settings)

    mock_create.assert_called_once_with(
        provider="groq",
        model="llama-test",
        api_key="test-key",
        timeout=30.0,
        max_retries=2,
    )

    assert result is mock_create.return_value