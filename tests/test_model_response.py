import pytest

from src.models.model_response import ModelResponse


def test_model_response_stores_normalized_data() -> None:
    response = ModelResponse(
        provider="openai",
        model="test-model",
        content="Test response",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=1.25,
        finish_reason="completed",
        response_id="response-123",
    )

    assert response.provider == "openai"
    assert response.content == "Test response"
    assert response.total_tokens == 15


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("total_tokens", -1),
        ("latency_seconds", -0.1),
    ],
)
def test_model_response_rejects_negative_values(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "provider": "openai",
        "model": "test-model",
        "content": "Test response",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "latency_seconds": 1.0,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        ModelResponse(**values)


def test_model_response_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="provider cannot be empty"):
        ModelResponse(
            provider=" ",
            model="test-model",
            content="Test response",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_seconds=1.0,
        )
