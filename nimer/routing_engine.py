"""Shared model fallback chain logic for sync and async Nimer clients."""
from __future__ import annotations

from ._constants import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET


def next_fallback_model(current_model: str) -> str | None:
    ordered = (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS)
    if current_model not in ordered:
        return MODEL_SONNET
    idx = ordered.index(current_model)
    if idx >= len(ordered) - 1:
        return None
    return ordered[idx + 1]


def fallback_chain(initial_model: str) -> tuple[str, ...]:
    chain = [initial_model]
    nxt = next_fallback_model(initial_model)
    while nxt is not None:
        chain.append(nxt)
        nxt = next_fallback_model(nxt)
    return tuple(chain)
