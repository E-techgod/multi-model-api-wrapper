# tests/test_llm_service.py

from unittest.mock import Mock

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse
from src.services.llm_service import LLMService

def test_llm_service_uses_injected_client() -> None:
    mock_client = Mock(spec=BaseLLMClient)

    expected_response = ModelResponse(
        content="Test response",
        provider="openai",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )

    mock_client.generate.return_value = expected_response

    service = LLMService(mock_client)

    result = service.generate(
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.2,
    )

    mock_client.generate.assert_called_once_with(
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.2,
    )

    assert result is expected_response