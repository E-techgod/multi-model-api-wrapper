# tests/test_llm_service.py
from unittest.mock import Mock

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse
from src.models.stream_event import LLMStreamEvent
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
        latency_seconds=1.0,
    )

    expected_events = [
        LLMStreamEvent(
            type="text_delta",
            delta="Test ",
            snapshot="Test ",
        ),
        LLMStreamEvent(
            type="response",
            snapshot="Test response",
            response=expected_response,
        ),
    ]

    mock_client.generate.return_value = iter(expected_events)
    mock_client.collect_response.return_value = expected_response

    service = LLMService(mock_client)

    result = list(
        service.generate(
            system_prompt="System prompt",
            user_prompt="User prompt",
            temperature=0.2,
        )
    )

    mock_client.generate.assert_called_once_with(
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.2,
    )

    assert result[0].delta == "Test "
    assert result[-1].response is expected_response

    collected = service.collect_response(
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.2,
    )

    mock_client.collect_response.assert_called_once_with(
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.2,
    )

    assert collected is expected_response
