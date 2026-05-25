"""Retry and circuit-breaker policy for Nimer SDK HTTP calls (VISION-5)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_ms: float = 200.0
    backoff_multiplier: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_ttl_secs: float = 60.0

    def __post_init__(self) -> None:
        self.max_attempts = max(1, int(self.max_attempts))
        self.initial_backoff_ms = max(0.0, float(self.initial_backoff_ms))
        self.backoff_multiplier = max(1.0, float(self.backoff_multiplier))
        self.circuit_breaker_threshold = max(1, int(self.circuit_breaker_threshold))
        self.circuit_breaker_ttl_secs = max(1.0, float(self.circuit_breaker_ttl_secs))

    def backoff_seconds(self, attempt: int) -> float:
        return (self.initial_backoff_ms / 1000.0) * (self.backoff_multiplier ** max(0, attempt - 1))


_CIRCUIT_STATE: dict[str, dict[str, Any]] = {}


def circuit_is_open(provider: str, policy: RetryPolicy) -> bool:
    state = _CIRCUIT_STATE.get(provider)
    if not state or not state.get("open"):
        return False
    opened_at = float(state.get("opened_at", 0.0))
    if time.monotonic() - opened_at >= policy.circuit_breaker_ttl_secs:
        state["open"] = False
        state["failures"] = 0
        return False
    return True


def record_provider_failure(provider: str, policy: RetryPolicy) -> None:
    state = _CIRCUIT_STATE.setdefault(provider, {"failures": 0, "open": False, "opened_at": 0.0})
    state["failures"] = int(state.get("failures", 0)) + 1
    if state["failures"] >= policy.circuit_breaker_threshold:
        state["open"] = True
        state["opened_at"] = time.monotonic()


def record_provider_success(provider: str) -> None:
    state = _CIRCUIT_STATE.get(provider)
    if state:
        state["failures"] = 0
        state["open"] = False
