"""Async usage example for Nimer SDK.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    export NIMER_API_KEY=nm_...    # optional
    python examples/async_usage.py
"""

from __future__ import annotations

import asyncio

from nimer import AsyncNimer


async def main() -> None:
    client = AsyncNimer()
    response = await client.messages.create(
        max_tokens=200,
        messages=[{"role": "user", "content": "Explain exponential backoff in 4 lines."}],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
