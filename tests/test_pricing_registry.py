# tests/pricing/test_pricing_registry.py

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.models.model_response import ModelResponse
from src.pricing.model_pricing import ModelPricing
from src.pricing.pricing_registry import PricingRegistry


def test_get_returns_none_for_unknown_model() -> None:
    result = PricingRegistry.get(
        provider="openai",
        model="unknown-model",
    )

    assert result is None


def test_require_rejects_unknown_model() -> None:
    with pytest.raises(
        ValueError,
        match="No pricing configured",
    ):
        PricingRegistry.require(
            provider="openai",
            model="unknown-model",
        )


@patch("src.models.model_response.PricingRegistry.get")
def test_model_response_calculates_cost_automatically(
    mock_get,
) -> None:
    mock_get.return_value = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
    )

    response = ModelResponse(
        provider="groq",
        model="test-model",
        content="Hello",
        input_tokens=500_000,
        output_tokens=100_000,
        total_tokens=600_000,
        latency_seconds=0.5,
    )

    assert response.input_cost == Decimal("0.500")
    assert response.output_cost == Decimal("0.500")
    assert response.total_cost == Decimal("1.000")


@patch("src.models.model_response.PricingRegistry.get")
def test_model_response_leaves_cost_empty_when_pricing_unknown(
    mock_get,
) -> None:
    mock_get.return_value = None

    response = ModelResponse(
        provider="unknown-provider",
        model="unknown-model",
        content="Hello",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_seconds=0.5,
    )

    assert response.input_cost is None
    assert response.output_cost is None
    assert response.total_cost is None


@patch("src.models.model_response.PricingRegistry.get")
def test_model_response_uses_requested_model_for_pricing(
    mock_get,
) -> None:
    mock_get.return_value = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("2.00"),
    )

    ModelResponse(
        provider="gemini",
        model="gemini-versioned-response-model",
        requested_model="gemini-configured-model",
        content="Hello",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_seconds=0.5,
    )

    mock_get.assert_called_once_with(
        provider="gemini",
        model="gemini-configured-model",
    )
