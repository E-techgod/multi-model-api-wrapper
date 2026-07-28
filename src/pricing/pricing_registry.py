# src/pricing/pricing_registry.py

from decimal import Decimal
from typing import ClassVar

from src.pricing.model_pricing import ModelPricing


class PricingRegistry:
    """Stores pricing by normalized provider and exact model ID."""

    _PRICES: ClassVar[dict[tuple[str, str], ModelPricing]] = {
        # Add only exact model IDs used by the application.
        #
        # Example:
        # (
        #     "provider",
        #     "exact-model-id",
        # ): ModelPricing(
        #     input_per_million=Decimal("0.00"),
        #     output_per_million=Decimal("0.00"),
        # ),
        (
            "groq",
            "openai/gpt-oss-20b",
        ): ModelPricing(
            input_per_million=Decimal("0.075"),
            output_per_million=Decimal("0.30"),
        ),
        (
            "gemini",
            "gemini-3.1-flash-lite",
        ): ModelPricing(
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("1.50"),
        ),
        (
            "openai",
            "gpt-4o-mini-2024-07-18",
        ): ModelPricing(
            input_per_million=Decimal("0.15"),
            output_per_million=Decimal("0.60"),
        ),
        (
            "anthropic",
            "claude-haiku-4-5-20251001",
        ): ModelPricing(
            input_per_million=Decimal("1.00"),
            output_per_million=Decimal("5.00"),
        ),
        (
            "groq",
            "llama-3.1-8b-instant",
        ): ModelPricing(
            input_per_million=Decimal("0.05"),
            output_per_million=Decimal("0.08"),
        ),
    }

    @classmethod
    def get(
        cls,
        *,
        provider: str,
        model: str,
    ) -> ModelPricing | None:
        normalized_provider = provider.strip().lower()
        normalized_model = model.strip().lower()

        if not normalized_provider:
            raise ValueError("provider cannot be empty")

        if not normalized_model:
            raise ValueError("model cannot be empty")

        return cls._PRICES.get((normalized_provider, normalized_model))

    @classmethod
    def require(
        cls,
        *,
        provider: str,
        model: str,
    ) -> ModelPricing:
        pricing = cls.get(
            provider=provider,
            model=model,
        )

        if pricing is None:
            raise ValueError(
                "No pricing configured for " f"provider={provider!r}, model={model!r}"
            )

        return pricing
