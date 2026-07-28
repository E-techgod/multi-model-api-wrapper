# tests/test_client_factory.py
import pytest
from unittest.mock import patch

from src.clients.anthropic import AnthropicClient
from src.clients.gemini import GeminiClient
from src.clients.groq import GroqClient
from src.clients.openai import OpenAIClient
from src.factory.client_factory import ClientFactory
from src.factory.providers import LLMProvider

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