"""Custom exceptions raised by the Nimer SDK."""


class NimerError(Exception):
    """Base exception for all Nimer SDK errors."""


class ConfigurationError(NimerError):
    """Raised when the SDK is misconfigured (e.g. missing API key)."""


class RoutingError(NimerError):
    """Raised when routing logic cannot select a valid model."""
