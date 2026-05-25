"""Shared sync/async client helpers for OptimizedClaude and AsyncNimer."""
from __future__ import annotations

import time
from typing import Any

from .exceptions import TrustGatewayError
from .logger import UsageLogger
from .pricing import estimate_savings


class _SecretStr(str):
    """API key holder that does not leak in repr/str."""

    def __repr__(self) -> str:
        return "'[REDACTED]'"

    def __str__(self) -> str:
        return "[REDACTED]"


def _as_secret(value: str | None) -> _SecretStr | None:
    return _SecretStr(value) if value else None


def _build_forward_kwargs(
    *,
    chosen_model: str,
    filtered_messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    fwd: dict[str, Any] = {
        "model": chosen_model,
        "messages": filtered_messages,
        **kwargs,
    }
    if system is not None:
        fwd["system"] = system
    return fwd


def _merge_trust_report(report: Any, input_issues: tuple[Any, ...], gateway: Any) -> Any:
    if not input_issues:
        return report
    merged_issues = tuple(input_issues) + tuple(report.issues)
    merged_score = gateway._score_issues(list(merged_issues))
    return report.__class__(
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


def _log_usage_safe(
    usage_logger: UsageLogger | None,
    *,
    requested_model: str | None,
    chosen_model: str,
    input_tokens: int,
    output_tokens: int,
    report_dict: dict[str, Any],
    auto_routed: bool,
    blocked_by_gateway: bool,
    fallback_attempts: int,
) -> None:
    if usage_logger is None:
        return
    try:
        savings = estimate_savings(
            actual_model=chosen_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        usage_logger.log_async(
            requested_model=requested_model,
            actual_model=chosen_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_savings_usd=savings,
            auto_routed=auto_routed,
        )
        usage_logger.log_trust_async(
            requested_model=requested_model,
            actual_model=chosen_model,
            report=report_dict,
            blocked_by_gateway=blocked_by_gateway,
            fallback_attempts=fallback_attempts,
        )
    except Exception:  # pragma: no cover - defensive
        pass


def _run_messages_create_sync(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    auto_route: bool,
    model: str | None,
    system: str | list[dict[str, Any]] | None,
    baseline_model: str,
    kwargs: dict[str, Any],
) -> Any:
    filtered_messages, input_issues, input_blocked = client._trust_gateway.filter_input(
        messages,
        system=system,
    )
    if input_blocked:
        raise TrustGatewayError("AI Quality & Trust Gateway blocked unsafe input prompt.")

    if auto_route:
        chosen_model = client._router.choose(messages, system=system)
    else:
        chosen_model = model or baseline_model

    forward_kwargs = _build_forward_kwargs(
        chosen_model=chosen_model,
        filtered_messages=filtered_messages,
        system=system,
        kwargs=kwargs,
    )

    report = None
    response = None
    input_tokens = 0
    output_tokens = 0
    blocked_by_gateway = False
    fallback_attempts = 0

    for candidate_model in client._fallback_chain(chosen_model):
        if candidate_model != chosen_model:
            fallback_attempts += 1
        forward_kwargs["model"] = candidate_model
        response, latency_ms = client._call_with_latency(forward_kwargs)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        try:
            report = client._trust_gateway.enforce(
                response,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if input_issues:
                report = _merge_trust_report(report, input_issues, client._trust_gateway)
            chosen_model = candidate_model
            break
        except TrustGatewayError:
            raise

    if report is None or response is None:
        raise TrustGatewayError("AI Quality & Trust Gateway blocked the response.")

    try:
        setattr(response, "_nimer_trust_report", report.to_dict())
    except Exception:  # pragma: no cover
        pass

    _log_usage_safe(
        client._usage_logger,
        requested_model=model,
        chosen_model=chosen_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        report_dict=report.to_dict(),
        auto_routed=(auto_route and model is None),
        blocked_by_gateway=blocked_by_gateway,
        fallback_attempts=fallback_attempts,
    )
    return response


async def _run_messages_create_async(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    auto_route: bool,
    model: str | None,
    system: str | list[dict[str, Any]] | None,
    baseline_model: str,
    kwargs: dict[str, Any],
) -> Any:
    filtered_messages, input_issues, input_blocked = client._trust_gateway.filter_input(
        messages,
        system=system,
    )
    if input_blocked:
        raise TrustGatewayError("AI Quality & Trust Gateway blocked unsafe input prompt.")

    if auto_route:
        chosen_model = client._router.choose(messages, system=system)
    else:
        chosen_model = model or baseline_model

    forward_kwargs = _build_forward_kwargs(
        chosen_model=chosen_model,
        filtered_messages=filtered_messages,
        system=system,
        kwargs=kwargs,
    )

    report = None
    response = None
    input_tokens = 0
    output_tokens = 0
    blocked_by_gateway = False
    fallback_attempts = 0

    for candidate_model in client._fallback_chain(chosen_model):
        if candidate_model != chosen_model:
            fallback_attempts += 1
        forward_kwargs["model"] = candidate_model
        started = time.perf_counter()
        response = await client._anthropic.messages.create(**forward_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        try:
            report = client._trust_gateway.enforce(
                response,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if input_issues:
                report = _merge_trust_report(report, input_issues, client._trust_gateway)
            chosen_model = candidate_model
            break
        except TrustGatewayError:
            raise

    if report is None or response is None:
        raise TrustGatewayError("AI Quality & Trust Gateway blocked the response.")

    try:
        setattr(response, "_nimer_trust_report", report.to_dict())
    except Exception:  # pragma: no cover
        pass

    _log_usage_safe(
        client._usage_logger,
        requested_model=model,
        chosen_model=chosen_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        report_dict=report.to_dict(),
        auto_routed=(auto_route and model is None),
        blocked_by_gateway=blocked_by_gateway,
        fallback_attempts=fallback_attempts,
    )
    return response
