# tests/pricing/test_cost_calculator.py

from decimal import Decimal

import pytest

from src.pricing.cost_calculator import calculate_usage_cost
from src.pricing.model_pricing import ModelPricing

def test_calculate_usage_cost() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
    )

    result = calculate_usage_cost(
        input_tokens=500_000,
        output_tokens=100_000,
        pricing=pricing,
    )

    assert result.input_cost == Decimal("0.500")
    assert result.output_cost == Decimal("0.500")
    assert result.total_cost == Decimal("1.000")

def test_calculate_usage_cost_with_zero_tokens() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
    )

    result = calculate_usage_cost(
        input_tokens=0,
        output_tokens=0,
        pricing=pricing,
    )

    assert result.input_cost == Decimal("0")
    assert result.output_cost == Decimal("0")
    assert result.total_cost == Decimal("0")

def test_rejects_negative_input_tokens() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
    )

    with pytest.raises(
        ValueError,
        match="input_tokens cannot be negative",
    ):
        calculate_usage_cost(
            input_tokens=-1,
            output_tokens=10,
            pricing=pricing,
        )

def test_rejects_negative_output_tokens() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
    )

    with pytest.raises(
        ValueError,
        match="output_tokens cannot be negative",
    ):
        calculate_usage_cost(
            input_tokens=10,
            output_tokens=-1,
            pricing=pricing,
        )