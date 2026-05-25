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
from .ai_quality_gateway import AIQualityTrustGateway, GatewayIssue, GatewayPolicy, GatewayReport
from ._version import __version__
from .client import AsyncNimer, OptimizedClaude
from .exceptions import (
    BudgetExceededError,
    ComplianceError,
    ConfigurationError,
    NimerError,
    ProviderError,
    RateLimitError,
    RoutingError,
    TrustGatewayError,
)
from .retry_policy import RetryPolicy
from .pricing import estimate_cost, estimate_savings
from .router import Router

# Drop-in alias: `from nimer import Anthropic` replaces `from anthropic import Anthropic`
Anthropic = OptimizedClaude

__all__ = [
    "Anthropic",
    "OptimizedClaude",
    "AsyncNimer",
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
    "TrustGatewayError",
    "ProviderError",
    "BudgetExceededError",
    "ComplianceError",
    "RateLimitError",
    "RetryPolicy",
    "AIQualityTrustGateway",
    "GatewayIssue",
    "GatewayPolicy",
    "GatewayReport",
    "__version__",
]
