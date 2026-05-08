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
import time
from typing import Any

from anthropic import Anthropic

from ._constants import BASELINE_MODEL, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET
from .ai_quality_gateway import AIQualityTrustGateway
from .exceptions import ConfigurationError, TrustGatewayError
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
        filtered_messages, input_issues, input_blocked = self._trust_gateway.filter_input(
            messages,
            system=system,
        )
        if input_blocked:
            raise TrustGatewayError("AI Quality & Trust Gateway blocked unsafe input prompt.")

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
            "messages": filtered_messages,
            **kwargs,
        }
        if system is not None:
            forward_kwargs["system"] = system

        report = None
        response = None
        latency_ms = 0.0
        input_tokens = 0
        output_tokens = 0
        blocked_by_gateway = False
        fallback_attempts = 0
        for candidate_model in self._fallback_chain(chosen_model):
            if candidate_model != chosen_model:
                fallback_attempts += 1
            forward_kwargs["model"] = candidate_model
            response, latency_ms = self._call_with_latency(forward_kwargs)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            try:
                report = self._trust_gateway.enforce(
                    response,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                if input_issues:
                    merged_issues = tuple(input_issues) + tuple(report.issues)
                    merged_score = self._trust_gateway._score_issues(list(merged_issues))
                    report = report.__class__(
                        is_valid=report.is_valid,
                        is_safe=report.is_safe,
                        is_biased=report.is_biased,
                        has_pii=report.has_pii or any(i.category == "pii" for i in input_issues),
                        is_toxic=report.is_toxic or any(i.category == "toxicity" for i in input_issues),
                        safety_score=min(report.safety_score, merged_score),
                        redacted_text=report.redacted_text,
                        input_tokens=report.input_tokens,
                        output_tokens=report.output_tokens,
                        total_tokens=report.total_tokens,
                        latency_ms=report.latency_ms,
                        issues=merged_issues,
                    )
                chosen_model = candidate_model
                break
            except TrustGatewayError:
                blocked_by_gateway = True
                continue

        if report is None or response is None:
            raise TrustGatewayError("AI Quality & Trust Gateway blocked all fallback models.")
        # Expose report for downstream observability without changing return shape.
        try:
            setattr(response, "_nimer_trust_report", report.to_dict())
        except Exception:  # pragma: no cover - some SDK response objects may be immutable
            pass

        # Best-effort metadata logging. Never let logging issues bubble
        # up — they'd undermine the "drop-in replacement" promise.
        if self._usage_logger is not None:
            try:
                report_dict = report.to_dict()
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
                self._usage_logger.log_trust_async(
                    requested_model=model,
                    actual_model=chosen_model,
                    report=report_dict,
                    blocked_by_gateway=blocked_by_gateway,
                    fallback_attempts=fallback_attempts,
                )
            except Exception:  # pragma: no cover - defensive
                pass

        return response

    def _call_with_latency(self, forward_kwargs: dict[str, Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        result = self._anthropic.messages.create(**forward_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return result, latency_ms

    @staticmethod
    def _next_fallback_model(current_model: str) -> str | None:
        ordered = (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS)
        if current_model not in ordered:
            return MODEL_SONNET
        idx = ordered.index(current_model)
        if idx >= len(ordered) - 1:
            return None
        return ordered[idx + 1]

    def _fallback_chain(self, initial_model: str) -> tuple[str, ...]:
        chain = [initial_model]
        nxt = self._next_fallback_model(initial_model)
        while nxt is not None:
            chain.append(nxt)
            nxt = self._next_fallback_model(nxt)
        return tuple(chain)


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
