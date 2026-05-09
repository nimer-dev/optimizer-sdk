"""Async streaming through Nimer's multi-provider gateway.

Run::

    NIMER_API_KEY=nm_... python examples/astream_usage.py
"""

from __future__ import annotations

import asyncio
import os

from nimer import AsyncNimer


async def main() -> None:
    client = AsyncNimer(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "unused-for-streaming"),
        nimer_api_key=os.environ["NIMER_API_KEY"],
    )

    messages = [
        {
            "role": "user",
            "content": "Write a haiku about cost-aware AI routing.",
        }
    ]

    final_event: dict | None = None
    async for event in client.astream(messages, max_tokens=200):
        if event["type"] == "delta":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            final_event = event
        elif event["type"] == "error":
            print(f"\n[error] {event['message']}")

    print()
    if final_event:
        print(
            f"[done] model={final_event['model']} "
            f"provider={final_event['provider']} "
            f"tok={final_event['input_tokens']}+{final_event['output_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
