"""Pricing data and cost/savings calculations.

NOTE: Prices are in USD per 1,000,000 tokens. Verify against Anthropic's
pricing page (https://www.anthropic.com/pricing) before each release —
prices change and we want the dashboard to show truthful numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._constants import (
    BASELINE_MODEL,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
)


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token pricing for one model."""

    input_per_million: float
    output_per_million: float


# Snapshot of public Anthropic pricing — keep in sync with anthropic.com/pricing.
PRICING: dict[str, ModelPrice] = {
    MODEL_HAIKU: ModelPrice(input_per_million=0.25, output_per_million=1.25),
    MODEL_SONNET: ModelPrice(input_per_million=3.00, output_per_million=15.00),
    MODEL_OPUS: ModelPrice(input_per_million=15.00, output_per_million=75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a single API call.

    Returns 0.0 for unknown models rather than raising — the SDK should
    never crash a user's workflow because a new model name appeared.
    """
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return (
        (input_tokens / 1_000_000) * price.input_per_million
        + (output_tokens / 1_000_000) * price.output_per_million
    )


def estimate_savings(
    actual_model: str,
    input_tokens: int,
    output_tokens: int,
    baseline_model: str = BASELINE_MODEL,
) -> float:
    """How much money this routed call saved vs. the naive baseline.

    Positive number = savings. Zero or negative = no savings (e.g. we
    routed up to Opus because the request needed it).
    """
    actual = estimate_cost(actual_model, input_tokens, output_tokens)
    baseline = estimate_cost(baseline_model, input_tokens, output_tokens)
    return baseline - actual
