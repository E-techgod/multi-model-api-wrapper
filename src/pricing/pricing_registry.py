# src/pricing/pricing_registry.py

from decimal import Decimal
from doctest import Example

from src.pricing.model_pricing import ModelPricing


class PricingRegistry:
    """Stores pricing by normalized provider and exact model ID."""

    _PRICES: dict[tuple[str, str], ModelPricing] = {
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
        ):ModelPricing(
                input_per_million=Decimal("0.01"),
                output_per_million=Decimal("0.01"),
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

        return cls._PRICES.get(
            (normalized_provider, normalized_model)
        )

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
                "No pricing configured for "
                f"provider={provider!r}, model={model!r}"
            )

        return pricing

    # With the updated graph and context reports generated. Fix this failed tests