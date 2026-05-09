"""Server-Sent Events parser for Nimer's `/v1/chat/completions` endpoint.

The endpoint is OpenAI-compatible: each SSE frame is a `chat.completion.chunk`
JSON object (or the literal `[DONE]` terminator). This module turns those raw
frames into a small, easy-to-consume event shape so SDK users don't have to
peek into OpenAI internals.

Internal — not part of the public API. Subject to change without notice.
"""
from __future__ import annotations

import json
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Event shape (documented in client.OptimizedClaude.stream)
# ---------------------------------------------------------------------------
#
# {"type": "delta", "content": "..."}
# {"type": "done",
#  "model": str, "provider": str | None,
#  "input_tokens": int, "output_tokens": int,
#  "auto_routed": bool, "latency_ms": float | None,
#  "finish_reason": str}
# {"type": "error",
#  "message": str, "provider": str | None,
#  "raw_error": str | None, "model": str | None}
#
# Anything we can't parse is silently dropped. The terminator [DONE] returns
# None so callers know to stop iterating.


def parse_sse_line(line: str) -> Optional[dict[str, Any]] | str:
    """Parse a single SSE line into either:

    - ``"DONE"`` for the terminator
    - a parsed event dict (one of the three shapes above)
    - ``None`` to skip (comment, blank, malformed JSON, empty delta with no
      finish_reason — we don't surface those to the user)
    """
    if not line:
        return None
    if line.startswith(":"):
        # SSE comment / heartbeat
        return None
    if not line.startswith("data:"):
        # Some proxies prefix lines with their own headers; ignore quietly.
        return None

    payload = line[5:].lstrip()
    if payload == "[DONE]":
        return "DONE"

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    return _translate_chunk(chunk)


def _translate_chunk(chunk: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Translate an OpenAI-style chunk into our simplified event shape."""
    # The streaming endpoint emits an `error` field on the chunk when a
    # provider call fails *after* SSE setup. Surface it as our error event
    # so user code can `if event["type"] == "error": ...`.
    if "error" in chunk and isinstance(chunk["error"], dict):
        err = chunk["error"]
        return {
            "type": "error",
            "message": err.get("message", "provider error"),
            "provider": err.get("provider"),
            "raw_error": err.get("raw_error"),
            "model": chunk.get("model"),
        }

    choices = chunk.get("choices") or []
    if not choices:
        return None

    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    content = delta.get("content")
    if content:
        return {"type": "delta", "content": content}

    if finish_reason is not None:
        usage = chunk.get("usage") or {}
        x_nimer = chunk.get("x_nimer") or {}
        return {
            "type": "done",
            "model": chunk.get("model") or "",
            "provider": x_nimer.get("provider"),
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "auto_routed": bool(x_nimer.get("auto_routed", False)),
            "latency_ms": x_nimer.get("latency_ms"),
            "finish_reason": finish_reason,
        }

    # Opening role-only frame ({"role": "assistant"}) — nothing to surface.
    return None


def build_payload(
    *,
    messages: list[dict[str, Any]],
    model: str | None,
    max_tokens: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build the request body for `/v1/chat/completions` with stream=True."""
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    if extra:
        # Don't let callers override stream/messages — silently keep ours.
        for k, v in extra.items():
            if k in {"messages", "stream"}:
                continue
            payload[k] = v
    return payload
