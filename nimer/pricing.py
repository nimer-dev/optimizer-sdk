"""Pricing data and cost/savings calculations.

Live prices: ``refresh_pricing_from_api()`` pulls ``GET /v1/catalog/pricing``
with a short TTL cache. Falls back to the embedded snapshot when offline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from ._constants import (
    BASELINE_MODEL,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
)

log = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.nimer.dev"
_CACHE_TTL_SECS = 300.0
_cached_at: float = 0.0
_cached_source: str = "embedded"


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token pricing for one model."""

    input_per_million: float
    output_per_million: float


# Snapshot — used when the catalog endpoint is unreachable.
PRICING: dict[str, ModelPrice] = {
    MODEL_HAIKU: ModelPrice(input_per_million=0.25, output_per_million=1.25),
    MODEL_SONNET: ModelPrice(input_per_million=3.00, output_per_million=15.00),
    MODEL_OPUS: ModelPrice(input_per_million=15.00, output_per_million=75.00),
}


def refresh_pricing_from_api(
    *,
    base_url: str = _DEFAULT_API_BASE,
    ttl_seconds: float = _CACHE_TTL_SECS,
    timeout: float = 5.0,
) -> bool:
    """Fetch catalog pricing into ``PRICING``. Returns True on success."""
    global _cached_at, _cached_source

    now = time.monotonic()
    if _cached_source == "api" and (now - _cached_at) < ttl_seconds:
        return True

    url = f"{base_url.rstrip('/')}/v1/catalog/pricing"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.debug("catalog pricing fetch failed: %s", exc)
        return False

    models = payload.get("models")
    if not isinstance(models, list):
        return False

    targets = {MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS}
    updated = 0
    for row in models:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if model_id not in targets:
            continue
        inp = row.get("input_per_million")
        out = row.get("output_per_million")
        if inp is None or out is None:
            continue
        PRICING[str(model_id)] = ModelPrice(
            input_per_million=float(inp),
            output_per_million=float(out),
        )
        updated += 1

    if updated == 0:
        return False
    _cached_at = now
    _cached_source = "api"
    return True


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a single API call."""
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
    """How much money this routed call saved vs. the naive baseline."""
    actual = estimate_cost(actual_model, input_tokens, output_tokens)
    baseline = estimate_cost(baseline_model, input_tokens, output_tokens)
    return baseline - actual
