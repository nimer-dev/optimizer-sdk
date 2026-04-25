"""Tests for cost and savings calculations."""

import pytest

from nimer import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    estimate_cost,
    estimate_savings,
)


def test_haiku_one_million_input_tokens():
    assert estimate_cost(MODEL_HAIKU, 1_000_000, 0) == pytest.approx(0.25)


def test_sonnet_mixed_io_cost():
    # 500K input + 100K output on Sonnet:
    # = (0.5 * $3) + (0.1 * $15) = $1.50 + $1.50 = $3.00
    assert estimate_cost(MODEL_SONNET, 500_000, 100_000) == pytest.approx(3.00)


def test_unknown_model_returns_zero():
    """Unknown models must not crash — we return 0 and move on."""
    assert estimate_cost("claude-future-9000", 1_000_000, 1_000_000) == 0.0


def test_savings_when_routing_down_from_sonnet_to_haiku():
    # Same workload on Haiku vs Sonnet baseline.
    saved = estimate_savings(
        actual_model=MODEL_HAIKU,
        input_tokens=1_000_000,
        output_tokens=200_000,
    )
    # Sonnet cost: 1.0 * $3 + 0.2 * $15 = $6.00
    # Haiku cost:  1.0 * $0.25 + 0.2 * $1.25 = $0.50
    # Savings:     $5.50
    assert saved == pytest.approx(5.50)


def test_savings_negative_when_routing_up_to_opus():
    """Routing up to Opus costs more than baseline — savings goes negative."""
    saved = estimate_savings(
        actual_model=MODEL_OPUS,
        input_tokens=100_000,
        output_tokens=10_000,
    )
    assert saved < 0
