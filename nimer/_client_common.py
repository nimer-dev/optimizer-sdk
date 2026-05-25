"""Shared sync/async client helpers for OptimizedClaude and AsyncNimer."""
from __future__ import annotations

import time
from typing import Any, AsyncIterator, Callable, Iterator

import httpx

from ._streaming import build_payload, parse_sse_line
from .exceptions import ConfigurationError, NimerError, TrustGatewayError, raise_for_http_status
from .logger import UsageLogger
from .pricing import estimate_savings
from .retry_policy import RetryPolicy
from ._http_helpers import arequest_with_policy, otel_trace_headers, request_with_policy

_CHAT_TIMEOUT_SECS = 30.0
_ULTRATHINK_TIMEOUT_SECS = 60.0
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


class _SecretStr(str):
    """API key holder that does not leak in repr (str value unchanged for auth)."""

    def __repr__(self) -> str:
        return "'[REDACTED]'"


def _as_secret(value: str | None) -> _SecretStr | None:
    return _SecretStr(value) if value else None


def _bearer_token(api_key: _SecretStr | None) -> str:
    if not api_key:
        raise ConfigurationError(
            "nimer_api_key is required (or set NIMER_API_KEY env var)."
        )
    return api_key


def _raise_for_stream_error(status: int, body: bytes) -> None:
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
    raise_for_http_status(
        status,
        f"Nimer streaming request failed (HTTP {status}): {msg}",
        detail=parsed,
    )


class _NimerBackendCore:
    """Shared Nimer HTTP API surface (ARCH-9 DRY base)."""

    _nimer_api_key: _SecretStr | None
    _nimer_base_url: str
    _retry_policy: RetryPolicy | None

    def _init_nimer_backend(
        self,
        *,
        nimer_api_key: str | None,
        base_url: str,
        retry_policy: RetryPolicy | None,
    ) -> None:
        self._nimer_api_key = _as_secret(nimer_api_key)
        self._nimer_base_url = base_url.rstrip("/")
        self._retry_policy = retry_policy

    def _nimer_json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_bearer_token(self._nimer_api_key)}",
            "Content-Type": "application/json",
            **otel_trace_headers(),
        }

    def _nimer_stream_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_bearer_token(self._nimer_api_key)}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    def _post_nimer_sync(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        timeout: float,
        http: httpx.Client,
    ) -> dict[str, Any]:
        url = f"{self._nimer_base_url}{path}"

        def _do() -> httpx.Response:
            return http.post(
                url, json=payload, headers=self._nimer_json_headers(), timeout=timeout
            )

        response = request_with_policy(self._retry_policy, "nimer", _do)
        return response.json()

    async def _post_nimer_async(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        timeout: float,
        http: httpx.AsyncClient,
    ) -> dict[str, Any]:
        url = f"{self._nimer_base_url}{path}"

        async def _do() -> httpx.Response:
            return await http.post(
                url, json=payload, headers=self._nimer_json_headers(), timeout=timeout
            )

        response = await arequest_with_policy(self._retry_policy, "nimer", _do)
        return response.json()

    def _iter_stream_sync(
        self,
        http: httpx.Client,
        *,
        messages: list[dict[str, Any]],
        model: str | None,
        max_tokens: int,
        extra: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        url = f"{self._nimer_base_url}/v1/chat/completions"
        payload = build_payload(
            messages=messages, model=model, max_tokens=max_tokens, extra=extra
        )
        with http.stream(
            "POST",
            url,
            json=payload,
            headers=self._nimer_stream_headers(),
            timeout=_STREAM_TIMEOUT,
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

    async def _iter_stream_async(
        self,
        http: httpx.AsyncClient,
        *,
        messages: list[dict[str, Any]],
        model: str | None,
        max_tokens: int,
        extra: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        url = f"{self._nimer_base_url}/v1/chat/completions"
        payload = build_payload(
            messages=messages, model=model, max_tokens=max_tokens, extra=extra
        )
        async with http.stream(
            "POST",
            url,
            json=payload,
            headers=self._nimer_stream_headers(),
            timeout=_STREAM_TIMEOUT,
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

    def _iter_stream_text_sync(
        self,
        events: Iterator[dict[str, Any]],
    ) -> Iterator[str]:
        for event in events:
            kind = event.get("type")
            if kind == "delta":
                yield event.get("content", "")
            elif kind == "error":
                raise NimerError(
                    event.get("message", "Provider error during stream.")
                )

    async def _iter_stream_text_async(
        self,
        events: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[str]:
        async for event in events:
            kind = event.get("type")
            if kind == "delta":
                yield event.get("content", "")
            elif kind == "error":
                raise NimerError(
                    event.get("message", "Provider error during stream.")
                )


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
