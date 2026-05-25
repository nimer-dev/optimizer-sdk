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
from ._streaming import build_payload, parse_sse_line
from .ai_quality_gateway import AIQualityTrustGateway
from .exceptions import ConfigurationError, NimerError, TrustGatewayError
from .logger import UsageLogger
from .router import Router
from .routing_engine import fallback_chain, next_fallback_model
from ._client_common import (
    _SecretStr,
    _as_secret,
    _run_messages_create_async,
    _run_messages_create_sync,
)


# Default timeout for Ultrathink fan-out + synthesis (slower than a single call).
_ULTRATHINK_TIMEOUT_SECS = 60.0
_CHAT_TIMEOUT_SECS = 30.0
# Streaming uses a long read timeout because we hold the connection open while
# the model generates tokens; the connect timeout stays short so DNS/TLS issues
# fail fast.
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


def _raise_for_stream_error(status: int, body: bytes) -> None:
    """Convert a non-2xx response from the streaming endpoint into a NimerError.

    The API returns structured `{"detail": ...}` JSON for 4xx/5xx; we forward
    the message verbatim so users see the same wording as the dashboard.
    """
    try:
        parsed = httpx.Response(status_code=status, content=body).json()
    except Exception:
        parsed = body.decode("utf-8", errors="replace")
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        if isinstance(detail, dict):
            msg = detail.get("message") or str(detail)
        else:
            msg = str(detail)
    else:
        msg = str(parsed)
    raise NimerError(f"Nimer streaming request failed (HTTP {status}): {msg}")


class OptimizedClaude:
    """Anthropic client with automatic, cost-aware model routing."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        nimer_api_key: str | None = None,
        *,
        base_url: str = "https://api.nimer.dev",
        router: Router | None = None,
        trust_gateway: AIQualityTrustGateway | None = None,
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
        self._nimer_api_key: _SecretStr | None = _as_secret(nimer_api_key)
        self._nimer_base_url = base_url.rstrip("/")
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
        return self._call_nimer_endpoint(
            path="/v1/chat",
            payload={"messages": messages, "mode": mode, **kwargs},
            timeout=_CHAT_TIMEOUT_SECS if mode == "auto" else _ULTRATHINK_TIMEOUT_SECS,
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
        return self._call_nimer_endpoint(
            path="/v1/ultrathink",
            payload={"messages": messages, **kwargs},
            timeout=_ULTRATHINK_TIMEOUT_SECS,
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
        if not self._nimer_api_key:
            raise ConfigurationError(
                "nimer_api_key is required for stream() "
                "(or set NIMER_API_KEY env var)."
            )

        url = f"{self._nimer_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._nimer_api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        payload = build_payload(
            messages=messages, model=model, max_tokens=max_tokens, extra=extra
        )
        http = self._get_http()
        with http.stream(
            "POST", url, json=payload, headers=headers, timeout=_STREAM_TIMEOUT
        ) as resp:
            if resp.status_code >= 400:
                _raise_for_stream_error(resp.status_code, resp.read())
            for line in resp.iter_lines():
                parsed = parse_sse_line(line)
                if parsed is None:
                    continue
                if parsed == "DONE":
                    return
                yield parsed  # type: ignore[misc]

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
        for event in self.stream(
            messages, model=model, max_tokens=max_tokens, **extra
        ):
            kind = event.get("type")
            if kind == "delta":
                yield event.get("content", "")
            elif kind == "error":
                raise NimerError(
                    event.get("message", "Provider error during stream.")
                )

    def _call_nimer_endpoint(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        if not self._nimer_api_key:
            raise ConfigurationError(
                "nimer_api_key is required for chat() / ultrathink() "
                "(or set NIMER_API_KEY env var)."
            )
        url = f"{self._nimer_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._nimer_api_key}",
            "Content-Type": "application/json",
        }
        http = self._get_http()
        response = http.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

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


class AsyncNimer:
    """Async drop-in for OptimizedClaude — full routing, Trust Gateway, and logging."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        nimer_api_key: str | None = None,
        *,
        base_url: str = "https://api.nimer.dev",
        router: Router | None = None,
        trust_gateway: AIQualityTrustGateway | None = None,
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
        self._nimer_api_key: _SecretStr | None = _as_secret(nimer_api_key)
        self._nimer_base_url = base_url.rstrip("/")
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
        return await self._acall_nimer_endpoint(
            path="/v1/chat",
            payload={"messages": messages, "mode": mode, **kwargs},
            timeout=_CHAT_TIMEOUT_SECS if mode == "auto" else _ULTRATHINK_TIMEOUT_SECS,
        )

    async def aultrathink(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`OptimizedClaude.ultrathink`. See its docstring."""
        return await self._acall_nimer_endpoint(
            path="/v1/ultrathink",
            payload={"messages": messages, **kwargs},
            timeout=_ULTRATHINK_TIMEOUT_SECS,
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
        if not self._nimer_api_key:
            raise ConfigurationError(
                "nimer_api_key is required for astream() "
                "(or set NIMER_API_KEY env var)."
            )

        url = f"{self._nimer_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._nimer_api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        payload = build_payload(
            messages=messages, model=model, max_tokens=max_tokens, extra=extra
        )
        http = await self._get_http()
        async with http.stream(
            "POST", url, json=payload, headers=headers, timeout=_STREAM_TIMEOUT
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                _raise_for_stream_error(resp.status_code, body)
            async for line in resp.aiter_lines():
                parsed = parse_sse_line(line)
                if parsed is None:
                    continue
                if parsed == "DONE":
                    return
                yield parsed  # type: ignore[misc]

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
                raise NimerError(
                    event.get("message", "Provider error during stream.")
                )

    async def _acall_nimer_endpoint(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        if not self._nimer_api_key:
            raise ConfigurationError(
                "nimer_api_key is required for achat() / aultrathink() "
                "(or set NIMER_API_KEY env var)."
            )
        url = f"{self._nimer_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._nimer_api_key}",
            "Content-Type": "application/json",
        }
        http = await self._get_http()
        response = await http.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _next_fallback_model(current_model: str) -> str | None:
        return next_fallback_model(current_model)

    def _fallback_chain(self, initial_model: str) -> tuple[str, ...]:
        return fallback_chain(initial_model)

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
