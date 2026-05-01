"""Nimer Optimizer — cut Claude API costs by routing to the cheapest model.

Quick start:

    from nimer import OptimizedClaude

    client = OptimizedClaude(
        anthropic_api_key="sk-ant-...",
        nimer_api_key="nm_...",  # optional — enables dashboard logging
    )

    response = client.messages.create(
        messages=[{"role": "user", "content": "Hello!"}],
    )

The SDK ships with a sensible default router. Pass a custom Router
instance if you want to tune thresholds.
"""

from ._constants import (
    BASELINE_MODEL,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    SUPPORTED_MODELS,
)
from ._version import __version__
from .client import OptimizedClaude
from .exceptions import ConfigurationError, NimerError, RoutingError
from .pricing import estimate_cost, estimate_savings
from .router import Router

# Drop-in alias: `from nimer import Anthropic` replaces `from anthropic import Anthropic`
Anthropic = OptimizedClaude

__all__ = [
    "Anthropic",
    "OptimizedClaude",
    "Router",
    "estimate_cost",
    "estimate_savings",
    "MODEL_HAIKU",
    "MODEL_SONNET",
    "MODEL_OPUS",
    "BASELINE_MODEL",
    "SUPPORTED_MODELS",
    "NimerError",
    "ConfigurationError",
    "RoutingError",
    "__version__",
]
