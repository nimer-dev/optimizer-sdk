"""Custom exceptions raised by the Nimer SDK."""
from __future__ import annotations

from typing import Any


class NimerError(Exception):
    """Base exception for all Nimer SDK errors."""

    status: int | None = None
    detail: Any = None

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class ConfigurationError(NimerError):
    """Raised when the SDK is misconfigured (e.g. missing API key)."""


class RoutingError(NimerError):
    """Raised when routing logic cannot select a valid model."""


class TrustGatewayError(NimerError):
    """Raised when AI Quality & Trust Gateway blocks a response."""


class ProviderError(NimerError):
    """Upstream LLM provider returned an error."""


class BudgetExceededError(NimerError):
    """Monthly or per-turn budget cap exceeded (HTTP 402)."""


class ComplianceError(NimerError):
    """Content blocked by halal/compliance layer (HTTP 451)."""


class RateLimitError(NimerError):
    """Rate limit exceeded (HTTP 429)."""


def raise_for_http_status(status: int, message: str, detail: Any = None) -> None:
    """Map API status codes to typed SDK exceptions."""
    if status == 402:
        raise BudgetExceededError(message, status=status, detail=detail)
    if status == 429:
        raise RateLimitError(message, status=status, detail=detail)
    if status == 451:
        raise ComplianceError(message, status=status, detail=detail)
    if 400 <= status < 500:
        raise ProviderError(message, status=status, detail=detail)
    if status >= 500:
        raise NimerError(message, status=status, detail=detail)
