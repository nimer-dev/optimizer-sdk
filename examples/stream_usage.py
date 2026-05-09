"""Streaming through Nimer's multi-provider gateway.

This uses Nimer's `/v1/chat/completions` endpoint, which auto-routes your
request to the cheapest connected provider (or honours an explicit `model`
hint) and streams tokens back over Server-Sent Events.

Set `NIMER_API_KEY` to a key from https://dashboard.nimer.dev/settings/api-keys.
You also need at least one provider connected at /settings/providers.

Run::

    NIMER_API_KEY=nm_... python examples/stream_usage.py

For Anthropic-only token streaming (no Nimer routing), see the
`messages.stream(...)` proxy on `OptimizedClaude` — that one is unchanged
and just forwards to the official Anthropic SDK.
"""

from __future__ import annotations

import os

from nimer import OptimizedClaude


def main() -> None:
    client = OptimizedClaude(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "unused-for-streaming"),
        nimer_api_key=os.environ["NIMER_API_KEY"],
    )

    messages = [
        {
            "role": "user",
            "content": "List 3 reasons a startup should adopt an AI gateway. "
            "Keep each point under 12 words.",
        }
    ]

    print("--- streaming response ---")
    final_event: dict | None = None
    for event in client.stream(messages, max_tokens=300):
        if event["type"] == "delta":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            final_event = event
        elif event["type"] == "error":
            print(f"\n[error] {event['message']}")

    print()  # newline after the streamed text
    if final_event:
        print(
            f"--- done · model={final_event['model']} "
            f"provider={final_event['provider']} "
            f"tokens={final_event['input_tokens']}+{final_event['output_tokens']} "
            f"auto_routed={final_event['auto_routed']} ---"
        )


if __name__ == "__main__":
    main()
