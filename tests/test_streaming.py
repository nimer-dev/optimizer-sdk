"""Tests for nimer._streaming — SSE parsing and request payload builder."""
from __future__ import annotations

import json

import pytest

from nimer import ConfigurationError, OptimizedClaude
from nimer._streaming import build_payload, parse_sse_line


def _data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}"


# ---------------------------------------------------------------------------
# parse_sse_line
# ---------------------------------------------------------------------------


def test_role_only_frame_is_skipped() -> None:
    line = _data(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )
    assert parse_sse_line(line) is None


def test_content_delta_returns_delta_event() -> None:
    line = _data(
        {
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]
        }
    )
    assert parse_sse_line(line) == {"type": "delta", "content": "Hello"}


def test_final_frame_returns_done_with_usage_and_x_nimer() -> None:
    line = _data(
        {
            "model": "claude-3-haiku-20240307",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 42,
                "total_tokens": 52,
            },
            "x_nimer": {
                "provider": "anthropic",
                "auto_routed": True,
                "latency_ms": 840,
            },
        }
    )
    event = parse_sse_line(line)
    assert event is not None and event != "DONE"
    assert event["type"] == "done"
    assert event["model"] == "claude-3-haiku-20240307"
    assert event["provider"] == "anthropic"
    assert event["input_tokens"] == 10
    assert event["output_tokens"] == 42
    assert event["auto_routed"] is True
    assert event["latency_ms"] == 840
    assert event["finish_reason"] == "stop"


def test_error_frame_returns_error_event() -> None:
    line = _data(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "error": {
                "message": "quota exceeded",
                "provider": "google",
                "raw_error": "429",
            },
        }
    )
    event = parse_sse_line(line)
    assert event is not None and event != "DONE"
    assert event["type"] == "error"
    assert event["provider"] == "google"
    assert "quota" in event["message"]
    assert event["raw_error"] == "429"


def test_done_terminator() -> None:
    assert parse_sse_line("data: [DONE]") == "DONE"


@pytest.mark.parametrize(
    "noise",
    [
        ": heartbeat",
        "",
        "event: ping",
        "id: 42",
        "data: {bad json}",
    ],
)
def test_noise_lines_are_skipped(noise: str) -> None:
    assert parse_sse_line(noise) is None


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_forces_stream_true_and_drops_messages_override() -> None:
    payload = build_payload(
        messages=[{"role": "user", "content": "hi"}],
        model=None,
        max_tokens=128,
        extra={
            "temperature": 0.7,
            "stream": False,  # caller cannot disable streaming here
            "messages": [{"role": "x", "content": "ignored"}],
        },
    )
    assert payload["stream"] is True
    assert "model" not in payload
    assert payload["temperature"] == 0.7
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_build_payload_includes_explicit_model() -> None:
    payload = build_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        max_tokens=256,
        extra={},
    )
    assert payload["model"] == "gpt-4o-mini"
    assert payload["max_tokens"] == 256


# ---------------------------------------------------------------------------
# OptimizedClaude.stream — error path before any network I/O
# ---------------------------------------------------------------------------


def test_stream_raises_configuration_error_without_nimer_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NIMER_API_KEY", raising=False)
    client = OptimizedClaude(
        anthropic_api_key="sk-ant-fake",
        nimer_api_key=None,
    )
    with pytest.raises(ConfigurationError, match="nimer_api_key"):
        # Pulling the first item is what triggers the check; just constructing
        # the generator wouldn't (sync generators are lazy).
        next(client.stream([{"role": "user", "content": "hi"}]))
