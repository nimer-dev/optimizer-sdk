"""Fire-and-forget usage logger.

PRIVACY: We log only metadata — token counts, model identifiers,
timestamps. Never the prompt or response content. This is a load-bearing
promise to users; do not break it.

Logging runs in a daemon thread with a hard timeout so a slow or
unreachable Nimer backend never delays the user's actual API call.
"""

from __future__ import annotations

import logging
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
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1/usage"
        self._timeout = timeout_seconds

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
        thread = threading.Thread(
            target=self._send,
            args=(payload,),
            daemon=True,
            name="nimer-usage-logger",
        )
        thread.start()

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            httpx.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Logging must never crash the host process. Whisper, don't shout.
            logger.debug("Nimer usage log failed: %s", exc)
