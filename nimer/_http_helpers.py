"""HTTP helpers: retries, circuit breaker, OTel trace headers (VISION-5/7)."""
from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from .exceptions import NimerError, raise_for_http_status
from .retry_policy import (
    RetryPolicy,
    circuit_is_open,
    record_provider_failure,
    record_provider_success,
)


def otel_trace_headers() -> dict[str, str]:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return {}
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return {}
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")
        flags = "01" if ctx.trace_flags.sampled else "00"
        return {"traceparent": f"00-{trace_id}-{span_id}-{flags}"}
    except Exception:
        return {}


def _parse_error_message(response: httpx.Response) -> str:
    try:
        parsed = response.json()
    except Exception:
        return response.text[:500] or f"HTTP {response.status_code}"
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)
        return str(detail)
    return str(parsed)


def request_with_policy(
    policy: RetryPolicy | None,
    provider: str,
    fn: Callable[[], httpx.Response],
) -> httpx.Response:
    """Run sync HTTP call with optional retry + circuit breaker."""
    if policy is None:
        response = fn()
        if response.status_code >= 400:
            raise_for_http_status(
                response.status_code,
                _parse_error_message(response),
                detail=response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
            )
        return response

    if circuit_is_open(provider, policy):
        raise NimerError(
            f"Circuit open for provider {provider!r}; retry after cooldown",
            status=503,
        )

    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = fn()
            if response.status_code >= 400:
                if response.status_code >= 500 and attempt < policy.max_attempts:
                    record_provider_failure(provider, policy)
                    time.sleep(policy.backoff_seconds(attempt))
                    continue
                raise_for_http_status(
                    response.status_code,
                    _parse_error_message(response),
                )
            record_provider_success(provider)
            return response
        except NimerError:
            record_provider_failure(provider, policy)
            raise
        except Exception as exc:
            last_exc = exc
            record_provider_failure(provider, policy)
            if attempt >= policy.max_attempts:
                break
            time.sleep(policy.backoff_seconds(attempt))
    raise NimerError(str(last_exc or "Request failed"))


async def arequest_with_policy(
    policy: RetryPolicy | None,
    provider: str,
    fn: Callable[[], Any],
) -> httpx.Response:
    import asyncio

    if policy is None:
        response = await fn()
        if response.status_code >= 400:
            raise_for_http_status(response.status_code, _parse_error_message(response))
        return response

    if circuit_is_open(provider, policy):
        raise NimerError(
            f"Circuit open for provider {provider!r}; retry after cooldown",
            status=503,
        )

    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = await fn()
            if response.status_code >= 400:
                if response.status_code >= 500 and attempt < policy.max_attempts:
                    record_provider_failure(provider, policy)
                    await asyncio.sleep(policy.backoff_seconds(attempt))
                    continue
                raise_for_http_status(
                    response.status_code,
                    _parse_error_message(response),
                )
            record_provider_success(provider)
            return response
        except NimerError:
            record_provider_failure(provider, policy)
            raise
        except Exception as exc:
            last_exc = exc
            record_provider_failure(provider, policy)
            if attempt >= policy.max_attempts:
                break
            await asyncio.sleep(policy.backoff_seconds(attempt))
    raise NimerError(str(last_exc or "Request failed"))
