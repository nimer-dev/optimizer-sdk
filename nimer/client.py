"""OptimizedClaude — a drop-in replacement for the Anthropic Python SDK.

Usage mirrors anthropic.Anthropic as closely as possible so users can
switch with one import change:

    from anthropic import Anthropic                  # before
    from nimer import OptimizedClaude as Anthropic   # after

The wrapper picks the cheapest Claude model that can handle each
request, then forwards everything else to the real Anthropic SDK.
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

from ._constants import BASELINE_MODEL
from .exceptions import ConfigurationError
from .logger import UsageLogger
from .pricing import estimate_savings
from .router import Router


class OptimizedClaude:
    """Anthropic client with automatic, cost-aware model routing."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        nimer_api_key: str | None = None,
        *,
        base_url: str = "https://api.nimer.dev",
        router: Router | None = None,
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
        self._usage_logger: UsageLogger | None = (
            UsageLogger(api_key=nimer_api_key, base_url=base_url)
            if nimer_api_key
            else None
        )

        # Mirror anthropic.Anthropic's `.messages.create(...)` shape.
        self.messages = _MessagesProxy(self)

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
        # Decide which model to actually call.
        # When auto_route=True the router always wins — the user's `model`
        # becomes the logged "requested_model" only. This is the core product
        # promise: same call, smarter (cheaper) model automatically.
        # Pass auto_route=False to pin a specific model.
        if auto_route:
            chosen_model = self._router.choose(messages, system=system)
        else:
            chosen_model = model or BASELINE_MODEL

        # Build the kwargs we forward to the real Anthropic SDK.
        forward_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            **kwargs,
        }
        if system is not None:
            forward_kwargs["system"] = system

        response = self._anthropic.messages.create(**forward_kwargs)

        # Best-effort metadata logging. Never let logging issues bubble
        # up — they'd undermine the "drop-in replacement" promise.
        if self._usage_logger is not None:
            try:
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                savings = estimate_savings(
                    actual_model=chosen_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self._usage_logger.log_async(
                    requested_model=model,
                    actual_model=chosen_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_savings_usd=savings,
                    auto_routed=(auto_route and model is None),
                )
            except Exception:  # pragma: no cover - defensive
                pass

        return response


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
