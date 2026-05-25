"""Type stubs for IDE completion (VISION-7)."""
from typing import Any, AsyncIterator, Iterator, overload

from .ai_quality_gateway import AIQualityTrustGateway
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
from .router import Router

class OptimizedClaude:
    def __init__(
        self,
        anthropic_api_key: str | None = ...,
        nimer_api_key: str | None = ...,
        *,
        base_url: str = ...,
        router: Router | None = ...,
        trust_gateway: AIQualityTrustGateway | None = ...,
        retry_policy: RetryPolicy | None = ...,
    ) -> None: ...
    messages: Any
    def ping(self) -> bool: ...
    def chat(self, messages: list[dict[str, Any]], mode: str = ..., **kwargs: Any) -> dict[str, Any]: ...
    def ultrathink(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        max_tokens: int = ...,
        **extra: Any,
    ) -> Iterator[dict[str, Any]]: ...
    def stream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        max_tokens: int = ...,
        **extra: Any,
    ) -> Iterator[str]: ...
    def feedback(
        self,
        *,
        task_type: str,
        model: str,
        quality_score: float,
        request_id: str | None = ...,
        success: bool = ...,
    ) -> dict[str, Any]: ...
    def close(self) -> None: ...
    def __enter__(self) -> OptimizedClaude: ...
    def __exit__(self, *exc: object) -> None: ...

class AsyncNimer:
    def __init__(
        self,
        anthropic_api_key: str | None = ...,
        nimer_api_key: str | None = ...,
        *,
        base_url: str = ...,
        router: Router | None = ...,
        trust_gateway: AIQualityTrustGateway | None = ...,
        retry_policy: RetryPolicy | None = ...,
    ) -> None: ...
    messages: Any
    async def ping(self) -> bool: ...
    async def achat(self, messages: list[dict[str, Any]], mode: str = ..., **kwargs: Any) -> dict[str, Any]: ...
    async def aultrathink(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...
    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        max_tokens: int = ...,
        **extra: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...
    async def astream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        max_tokens: int = ...,
        **extra: Any,
    ) -> AsyncIterator[str]: ...
    async def feedback(
        self,
        *,
        task_type: str,
        model: str,
        quality_score: float,
        request_id: str | None = ...,
        success: bool = ...,
    ) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> AsyncNimer: ...
    async def __aexit__(self, *exc: object) -> None: ...
