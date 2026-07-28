from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.pricing.cost_calculator import calculate_usage_cost
from src.pricing.pricing_registry import PricingRegistry


@dataclass
class ModelResponse:
    provider: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float

    requested_model: str | None = None

    finish_reason: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    raw_response: Any = None

    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    total_cost: Decimal | None = None

    def __post_init__(self) -> None:
        self._validate_usage()
        self._calculate_costs()

    def _validate_usage(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")

        if not self.model.strip():
            raise ValueError("model cannot be empty")

        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        if self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")

        if self.total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")

        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")

    def _calculate_costs(self) -> None:
        pricing_model = self.requested_model or self.model

        pricing = PricingRegistry.get(
            provider=self.provider,
            model=pricing_model,
        )

        if pricing is None:
            return

        usage_cost = calculate_usage_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            pricing=pricing,
        )

        self.input_cost = usage_cost.input_cost
        self.output_cost = usage_cost.output_cost
        self.total_cost = usage_cost.total_cost
