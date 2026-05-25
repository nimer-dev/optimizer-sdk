"""OptimizedClaude — a drop-in replacement for the Anthropic Python SDK.

Usage mirrors anthropic.Anthropic as closely as possible so users can
switch with one import change:

    from anthropic import Anthropic                  # before
    from nimer import OptimizedClaude as Anthropic   # after

The wrapper picks the cheapest Claude model that can handle each
request, then forwards everything else to the real Anthropic SDK.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, AsyncIterator, Iterator

import httpx
from anthropic import Anthropic, AsyncAnthropic

from ._constants import BASELINE_MODEL, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET
from .ai_quality_gateway import AIQualityTrustGateway
from .exceptions import ConfigurationError, TrustGatewayError
from .retry_policy import RetryPolicy
from .logger import UsageLogger
from .router import Router
from .routing_engine import fallback_chain, next_fallback_model
from ._client_common import (
    _CHAT_TIMEOUT_SECS,
    _NimerBackendCore,
    _ULTRATHINK_TIMEOUT_SECS,
    _run_messages_create_async,
    _run_messages_create_sync,
)


class OptimizedClaude(_NimerBackendCore):
    """Anthropic client with automatic, cost-aware model routing."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        nimer_api_key: str | None = None,
        *,
        base_url: str = "https://api.nimer.dev",
        router: Router | None = None,
        trust_gateway: AIQualityTrustGateway | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        # Fall back to env vars so users can keep the same setup as the
        # vanilla anthropic SDK (ANTHROPIC_API_KEY).
        anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ConfigurationError(
                "anthropic_api_key is required (or set ANTHROPIC_API_KEY env var)."
            )

        nimer_api_key = nimer_api_key or os.getenv("NIMER_API_KEY")

        self._anthropic = Anthropic(api_key=anthropic_api_key)
        self._router = router or Router()
        self._trust_gateway = trust_gateway or AIQualityTrustGateway()
        self._init_nimer_backend(
            nimer_api_key=nimer_api_key,
            base_url=base_url,
            retry_policy=retry_policy,
        )
        self._http: httpx.Client | None = None
        self._http_lock = threading.Lock()
        self._usage_logger: UsageLogger | None = (
            UsageLogger(api_key=nimer_api_key, base_url=base_url)
            if nimer_api_key
            else None
        )

        # Mirror anthropic.Anthropic's `.messages.create(...)` shape.
        self.messages = _MessagesProxy(self)

    def __repr__(self) -> str:
        return f"OptimizedClaude(base_url={self._nimer_base_url!r})"

    # Context-manager support so callers can do `with OptimizedClaude(...)`
    # and get deterministic socket cleanup on exit.
    def __enter__(self) -> "OptimizedClaude":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Release the persistent httpx client. Safe to call multiple times."""
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None

    def _get_http(self) -> httpx.Client:
        """Lazily create and reuse a single httpx.Client per instance."""
        with self._http_lock:
            if self._http is None or self._http.is_closed:
                self._http = httpx.Client(
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
            return self._http

    def ping(self) -> bool:
        """Warm TCP/TLS and verify API reachability."""
        try:
            http = self._get_http()
            response = http.get(
                f"{self._nimer_base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Multi-provider routing — calls the Nimer backend, NOT Anthropic
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        mode: str = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send messages to Nimer's smart router.

        Args:
            messages: list of ``{"role": "user/assistant", "content": "..."}`` dicts
            mode: ``"auto"`` (default) routes to the cheapest suitable model
                  among the user's connected providers; ``"ultrathink"`` queries
                  ALL connected providers and synthesizes the best answer.

        Example::

            response = client.chat([{"role": "user", "content": "Hello"}])
            response = client.chat(messages, mode="ultrathink")

        Returns the full JSON response from /v1/chat. Requires ``nimer_api_key``
        to be configured (raises :class:`ConfigurationError` otherwise).
        """
        return self._post_nimer_sync(
            path="/v1/chat",
            payload={"messages": messages, "mode": mode, **kwargs},
            timeout=_CHAT_TIMEOUT_SECS if mode == "auto" else _ULTRATHINK_TIMEOUT_SECS,
            http=self._get_http(),
        )

    def ultrathink(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """⚡ Ultrathink: query all connected AI providers simultaneously.

        Returns a synthesized answer combining the best insights from each
        provider's best-quality model. Requires 2+ providers connected in
        your Nimer dashboard.

        Example::

            response = client.ultrathink([
                {"role": "user", "content": "Best pricing strategy for B2B SaaS?"},
            ])
            print(response["content"])              # synthesized answer
            print(response["providers_used"])       # ["anthropic", "openai", ...]
            print(response["individual_responses"]) # raw response per provider
        """
        return self._post_nimer_sync(
            path="/v1/ultrathink",
            payload={"messages": messages, **kwargs},
            timeout=_ULTRATHINK_TIMEOUT_SECS,
            http=self._get_http(),
        )

    # ------------------------------------------------------------------
    # Streaming via Nimer's OpenAI-compatible /v1/chat/completions endpoint
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream a chat completion through Nimer (multi-provider, auto-routed).

        Args:
            messages: list of ``{"role": "user/assistant/system", "content": "..."}``
            model: explicit model ID from the catalog (e.g. ``"gpt-4o-mini"``,
                   ``"claude-3-haiku-20240307"``, ``"gemini/gemini-2.5-flash"``).
                   Omit to let Nimer auto-route to the cheapest model that fits.
            max_tokens: cap on generated tokens (1..16000). Default 2048.
            **extra: any other OpenAI-style fields (``temperature``, ``top_p``,
                     etc.) — forwarded transparently and ignored if a provider
                     doesn't support them.

        Yields event dicts with one of three shapes:

            ``{"type": "delta", "content": "..."}``
                Incremental text — most events look like this.
            ``{"type": "done",  "model": "...", "provider": "...",``
                ``"input_tokens": int, "output_tokens": int,``
                ``"auto_routed": bool, "latency_ms": float | None,``
                ``"finish_reason": "stop" | ...}``
                Final event after the model is finished.
            ``{"type": "error", "message": "...", "provider": "...",``
                ``"raw_error": "...", "model": "..."}``
                The provider failed mid-stream (rare). Iteration ends after.

        Example::

            for event in client.stream([{"role": "user", "content": "Hi"}]):
                if event["type"] == "delta":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    print(f"\\n[{event['model']}] "
                          f"{event['input_tokens']}+{event['output_tokens']} tok")
        """
        yield from self._iter_stream_sync(
            self._get_http(),
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            extra=extra,
        )

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> Iterator[str]:
        """Convenience wrapper around :meth:`stream` that yields raw text only.

        Skips ``done`` events and surfaces ``error`` events as :class:`NimerError`.

        Example::

            for token in client.stream_text(messages):
                print(token, end="", flush=True)
        """
        yield from self._iter_stream_text_sync(
            self.stream(
                messages, model=model, max_tokens=max_tokens, **extra
            )
        )

    # ------------------------------------------------------------------
    # Internal — used by the messages proxy
    # ------------------------------------------------------------------

    def _create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return _run_messages_create_sync(
            self,
            messages=messages,
            auto_route=auto_route,
            model=model,
            system=system,
            baseline_model=BASELINE_MODEL,
            kwargs=kwargs,
        )

    def feedback(
        self,
        *,
        task_type: str,
        model: str,
        quality_score: float,
        request_id: str | None = None,
        success: bool = True,
    ) -> dict[str, Any]:
        return self._post_nimer_sync(
            path="/v1/feedback/routing",
            payload={
                "task_type": task_type,
                "model": model,
                "quality_score": quality_score,
                "request_id": request_id,
                "success": success,
            },
            timeout=10.0,
            http=self._get_http(),
        )

    def _call_with_latency(self, forward_kwargs: dict[str, Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        result = self._anthropic.messages.create(**forward_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return result, latency_ms

    @staticmethod
    def _next_fallback_model(current_model: str) -> str | None:
        return next_fallback_model(current_model)

    def _fallback_chain(self, initial_model: str) -> tuple[str, ...]:
        return fallback_chain(initial_model)


class _MessagesProxy:
    """Mirrors `anthropic.Anthropic.messages` so user code stays identical."""

    def __init__(self, client: OptimizedClaude) -> None:
        self._client = client

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._client._create_message(
            messages=messages,
            auto_route=auto_route,
            model=model,
            system=system,
            **kwargs,
        )

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        filtered_messages, _, blocked = self._client._trust_gateway.filter_input(
            messages,
            system=system,
        )
        if blocked:
            raise TrustGatewayError(
                "AI Quality & Trust Gateway blocked unsafe input prompt."
            )
        chosen_model = (
            self._client._router.choose(filtered_messages, system=system)
            if auto_route
            else (model or BASELINE_MODEL)
        )
        forward_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": filtered_messages,
            **kwargs,
        }
        if system is not None:
            forward_kwargs["system"] = system
        return self._client._anthropic.messages.stream(**forward_kwargs)


class AsyncNimer(_NimerBackendCore):
    """Async drop-in for OptimizedClaude — full routing, Trust Gateway, and logging."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        nimer_api_key: str | None = None,
        *,
        base_url: str = "https://api.nimer.dev",
        router: Router | None = None,
        trust_gateway: AIQualityTrustGateway | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ConfigurationError(
                "anthropic_api_key is required (or set ANTHROPIC_API_KEY env var)."
            )
        nimer_api_key = nimer_api_key or os.getenv("NIMER_API_KEY")
        self._anthropic = AsyncAnthropic(api_key=anthropic_api_key)
        self._router = router or Router()
        self._trust_gateway = trust_gateway or AIQualityTrustGateway()
        self._init_nimer_backend(
            nimer_api_key=nimer_api_key,
            base_url=base_url,
            retry_policy=retry_policy,
        )
        self._http: httpx.AsyncClient | None = None
        self._http_lock = asyncio.Lock()
        self._usage_logger: UsageLogger | None = (
            UsageLogger(api_key=nimer_api_key, base_url=base_url) if nimer_api_key else None
        )
        self.messages = _AsyncMessagesProxy(self)

    def __repr__(self) -> str:
        return f"AsyncNimer(base_url={self._nimer_base_url!r})"

    async def __aenter__(self) -> "AsyncNimer":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the persistent async httpx client. Safe to call multiple times."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    async def _get_http(self) -> httpx.AsyncClient:
        async with self._http_lock:
            if self._http is None or self._http.is_closed:
                self._http = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
            return self._http

    async def ping(self) -> bool:
        """Warm TCP/TLS and verify API reachability."""
        try:
            http = await self._get_http()
            response = await http.get(
                f"{self._nimer_base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Async multi-provider routing — calls the Nimer backend, NOT Anthropic
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, Any]],
        mode: str = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`OptimizedClaude.chat`. See its docstring."""
        return await self._post_nimer_async(
            path="/v1/chat",
            payload={"messages": messages, "mode": mode, **kwargs},
            timeout=_CHAT_TIMEOUT_SECS if mode == "auto" else _ULTRATHINK_TIMEOUT_SECS,
            http=await self._get_http(),
        )

    async def aultrathink(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`OptimizedClaude.ultrathink`. See its docstring."""
        return await self._post_nimer_async(
            path="/v1/ultrathink",
            payload={"messages": messages, **kwargs},
            timeout=_ULTRATHINK_TIMEOUT_SECS,
            http=await self._get_http(),
        )

    # ------------------------------------------------------------------
    # Async streaming via Nimer's /v1/chat/completions endpoint
    # ------------------------------------------------------------------

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async version of :meth:`OptimizedClaude.stream`. See that docstring
        for the full event shape.

        Example::

            async for event in client.astream([{"role": "user", "content": "Hi"}]):
                if event["type"] == "delta":
                    print(event["content"], end="", flush=True)
        """
        async for event in self._iter_stream_async(
            await self._get_http(),
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            extra=extra,
        ):
            yield event

    async def astream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Async equivalent of :meth:`OptimizedClaude.stream_text`."""
        async for event in self.astream(
            messages, model=model, max_tokens=max_tokens, **extra
        ):
            kind = event.get("type")
            if kind == "delta":
                yield event.get("content", "")
            elif kind == "error":
                from .exceptions import NimerError

                raise NimerError(
                    event.get("message", "Provider error during stream.")
                )

    @staticmethod
    def _next_fallback_model(current_model: str) -> str | None:
        return next_fallback_model(current_model)

    def _fallback_chain(self, initial_model: str) -> tuple[str, ...]:
        return fallback_chain(initial_model)

    async def feedback(
        self,
        *,
        task_type: str,
        model: str,
        quality_score: float,
        request_id: str | None = None,
        success: bool = True,
    ) -> dict[str, Any]:
        return await self._post_nimer_async(
            path="/v1/feedback/routing",
            payload={
                "task_type": task_type,
                "model": model,
                "quality_score": quality_score,
                "request_id": request_id,
                "success": success,
            },
            timeout=10.0,
            http=await self._get_http(),
        )

    async def _create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await _run_messages_create_async(
            self,
            messages=messages,
            auto_route=auto_route,
            model=model,
            system=system,
            baseline_model=BASELINE_MODEL,
            kwargs=kwargs,
        )


class _AsyncMessagesProxy:
    def __init__(self, client: AsyncNimer) -> None:
        self._client = client

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._client._create_message(
            messages=messages,
            auto_route=auto_route,
            model=model,
            system=system,
            **kwargs,
        )

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        auto_route: bool = True,
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        filtered_messages, _, blocked = self._client._trust_gateway.filter_input(
            messages,
            system=system,
        )
        if blocked:
            raise TrustGatewayError(
                "AI Quality & Trust Gateway blocked unsafe input prompt."
            )
        chosen_model = (
            self._client._router.choose(filtered_messages, system=system)
            if auto_route
            else (model or BASELINE_MODEL)
        )
        forward_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": filtered_messages,
            **kwargs,
        }
        if system is not None:
            forward_kwargs["system"] = system
        return self._client._anthropic.messages.stream(**forward_kwargs)
