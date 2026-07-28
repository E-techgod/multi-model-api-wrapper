# src/pricing/model_pricing.py

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """Token pricing for one provider/model combination."""

    input_per_million: Decimal
    output_per_million: Decimal

    def __post_init__(self) -> None:
        if self.input_per_million < 0:
            raise ValueError("input_per_million cannot be negative")

        if self.output_per_million < 0:
            raise ValueError("output_per_million cannot be negative")
