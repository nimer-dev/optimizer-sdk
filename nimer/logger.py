"""Fire-and-forget usage logger.

PRIVACY: We log only metadata — token counts, model identifiers,
timestamps. Never the prompt or response content. This is a load-bearing
promise to users; do not break it.

Logging runs in a daemon thread with a hard timeout so a slow or
unreachable Nimer backend never delays the user's actual API call.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger("nimer.logger")


class UsageLogger:
    """Send usage metadata to the Nimer backend asynchronously."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.nimer.dev",
        timeout_seconds: float = 2.0,
        max_queue_size: int = 1024,
    ) -> None:
        self._api_key = api_key
        root = base_url.rstrip("/")
        self._usage_url = f"{root}/v1/usage"
        self._trust_url = f"{root}/v1/trust"
        self._timeout = timeout_seconds
        self._http = httpx.Client(timeout=timeout_seconds)
        self._queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=max_queue_size)
        self._worker = threading.Thread(
            target=self._run_worker,
            daemon=True,
            name="nimer-logger-worker",
        )
        self._worker.start()

    def log_async(
        self,
        *,
        requested_model: str | None,
        actual_model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_savings_usd: float,
        auto_routed: bool,
    ) -> None:
        """Schedule a log POST in a background thread. Never raises."""
        payload = {
            "ts": time.time(),
            "requested_model": requested_model,
            "actual_model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_savings_usd": round(estimated_savings_usd, 6),
            "auto_routed": auto_routed,
        }
        self._enqueue("usage", payload)

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            self._http.post(
                self._usage_url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Logging must never crash the host process. Whisper, don't shout.
            logger.debug("Nimer usage log failed: %s", exc)

    def log_trust_async(
        self,
        *,
        requested_model: str | None,
        actual_model: str,
        report: dict[str, Any],
        blocked_by_gateway: bool,
        fallback_attempts: int,
    ) -> None:
        payload = {
            "ts": time.time(),
            "requested_model": requested_model,
            "actual_model": actual_model,
            "is_valid": bool(report.get("is_valid", True)),
            "is_safe": bool(report.get("is_safe", True)),
            "is_biased": bool(report.get("is_biased", False)),
            "has_pii": bool(report.get("has_pii", False)),
            "is_toxic": bool(report.get("is_toxic", False)),
            "safety_score": float(report.get("safety_score", 100.0) or 100.0),
            "blocked_by_gateway": blocked_by_gateway,
            "input_tokens": int(report.get("input_tokens", 0) or 0),
            "output_tokens": int(report.get("output_tokens", 0) or 0),
            "total_tokens": int(report.get("total_tokens", 0) or 0),
            "latency_ms": float(report.get("latency_ms", 0.0) or 0.0),
            "fallback_attempts": max(fallback_attempts, 0),
            "issues": report.get("issues", []),
        }
        self._enqueue("trust", payload)

    def _send_trust(self, payload: dict[str, Any]) -> None:
        try:
            self._http.post(
                self._trust_url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Nimer trust log failed: %s", exc)

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            logger.debug("Nimer logger queue full, dropping %s event", kind)

    def _run_worker(self) -> None:
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "trust":
                    self._send_trust(payload)
                else:
                    self._send(payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Nimer logger worker error: %s", exc)
            finally:
                self._queue.task_done()
