# src/pricing/cost_calculator.py

from dataclasses import dataclass
from decimal import Decimal

from src.pricing.model_pricing import ModelPricing

TOKENS_PER_MILLION = Decimal(1000000)


@dataclass(frozen=True)
class UsageCost:
    """Calculated USD cost for one model response."""

    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


def calculate_usage_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing,
) -> UsageCost:
    """Calculate model usage cost in USD."""

    if input_tokens < 0:
        raise ValueError("input_tokens cannot be negative")

    if output_tokens < 0:
        raise ValueError("output_tokens cannot be negative")

    input_cost = Decimal(input_tokens) / TOKENS_PER_MILLION * pricing.input_per_million

    output_cost = (
        Decimal(output_tokens) / TOKENS_PER_MILLION * pricing.output_per_million
    )

    return UsageCost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )
