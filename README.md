# Nimer Optimizer

> Cut your Claude API bill by 60% with intelligent model routing.

A drop-in replacement for the Anthropic Python SDK that routes each request to the cheapest Claude model that can handle it — without sacrificing quality.

[![Twitter](https://img.shields.io/badge/twitter-%40bynimer-1DA1F2)](https://twitter.com/bynimer)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://nimer.dev)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The problem

Most developers default to Sonnet — or worse, Opus — for every Claude API call. The price difference is dramatic:

| Model  | Input ($/1M tokens) | Output ($/1M tokens) | Relative cost |
|--------|---------------------|----------------------|---------------|
| Haiku  | $0.25               | $1.25                | 1×            |
| Sonnet | $3.00               | $15.00               | 12×           |
| Opus   | $15.00              | $75.00               | 60×           |

Most tasks — classification, lookups, simple Q&A, short summaries — work just as well on Haiku. The result is monthly bills that should be $80 turning into $400.

## The fix

Nimer Optimizer analyzes each prompt — length, content type, code presence — and routes it to the cheapest model that can handle it. Same quality, fraction of the cost.

## Quick start

```bash
pip install nimer
```

```python
from nimer import OptimizedClaude

client = OptimizedClaude(
    anthropic_api_key="sk-ant-...",
    nimer_api_key="nm_...",  # optional — enables dashboard logging
)

response = client.messages.create(
    max_tokens=512,
    messages=[{"role": "user", "content": "Translate 'good morning' to Arabic."}],
)
```

Three lines of config. Automatic routing. Real savings.

### Ready-to-copy snippets

#### Sync

```python
from nimer import OptimizedClaude

client = OptimizedClaude()
response = client.messages.create(
    max_tokens=256,
    messages=[{"role": "user", "content": "Summarize this in 3 bullets."}],
)
print(response.content[0].text)
```

#### Async

```python
import asyncio
from nimer import AsyncNimer

async def main() -> None:
    client = AsyncNimer()
    response = await client.messages.create(
        max_tokens=256,
        messages=[{"role": "user", "content": "Explain retries in one paragraph."}],
    )
    print(response.content[0].text)

asyncio.run(main())
```

#### Stream (Anthropic passthrough)

```python
from nimer import OptimizedClaude

client = OptimizedClaude()
with client.messages.stream(
    max_tokens=300,
    messages=[{"role": "user", "content": "Write a launch announcement thread."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

#### Stream through Nimer (multi-provider, auto-routed)

Streams from whichever provider you've connected — Claude, GPT, Gemini, DeepSeek, etc. — using Nimer's OpenAI-compatible `/v1/chat/completions` endpoint:

```python
from nimer import OptimizedClaude

client = OptimizedClaude(nimer_api_key="nm_...")
for event in client.stream(
    messages=[{"role": "user", "content": "Write a launch announcement thread."}],
    max_tokens=300,
    # model="claude-3-haiku-20240307",  # optional — pin a specific model
):
    if event["type"] == "delta":
        print(event["content"], end="", flush=True)
    elif event["type"] == "done":
        print(f"\n[{event['model']}] {event['input_tokens']}+{event['output_tokens']} tok")
```

For an even simpler flow, use `client.stream_text(...)` (yields raw token strings, raises on error). Async variants live on `AsyncNimer` as `astream(...)` / `astream_text(...)`.

#### Tools

```python
from nimer import OptimizedClaude

client = OptimizedClaude()
response = client.messages.create(
    max_tokens=400,
    tools=[
        {
            "name": "get_weather",
            "description": "Fetch weather by city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What's weather in Riyadh?"}],
)
print(response.content)
```

### Examples folder

- `examples/basic_usage.py` — minimal sync flow
- `examples/async_usage.py` — async flow
- `examples/stream_usage.py` — streaming flow
- `examples/tools_usage.py` — tool-use flow
- `examples/routing_demo.py` — cost/savings demo across prompt types

## Migrating from `anthropic`

Change one import:

```diff
- from anthropic import Anthropic
+ from nimer import OptimizedClaude as Anthropic
```

Everything else — `messages.create`, system prompts, streaming, tool use, multi-modal content — works identically.

## How routing works

Deterministic, explainable rules. No AI deciding which AI to use.

| Condition                                    | Routed to |
|----------------------------------------------|-----------|
| Total input > 5,000 chars                    | Sonnet    |
| Code request + last message > 500 chars      | Opus      |
| Last message < 200 chars                     | Haiku     |
| Everything else                              | Sonnet    |

You can:

- **Override per call:** `client.messages.create(model="claude-opus-4-7", auto_route=False, ...)`
- **Tune thresholds:** pass a custom `Router` instance with different limits
- **Inspect decisions:** every routing call is logged to your dashboard with the chosen model and estimated savings

## What we log (and what we don't)

If you set `nimer_api_key`, the SDK sends:

- Token counts (input + output)
- Which model was selected
- Estimated savings (USD)
- Timestamp

The SDK never sends:

- Your prompts
- Your responses
- Your Anthropic API key

This is enforced in code, not policy. Read [`nimer/logger.py`](nimer/logger.py) — it's 80 lines.

## Pricing

| Plan   | Price    | Includes                                                          |
|--------|----------|-------------------------------------------------------------------|
| Free   | $0/mo    | 1,000 requests/month, basic routing, 7-day analytics              |
| Pro    | $29/mo   | 100K requests, advanced routing, 90-day analytics, budget alerts  |
| Scale  | $99/mo   | Unlimited requests, custom routing rules, team features           |

The SDK itself is open source and free forever. The paid tiers add the dashboard, analytics, and managed routing rules.

## Status

🚧 **Alpha** — public launch in 6 weeks.

- [x] Core SDK with rule-based routing
- [x] Cost & savings calculations
- [x] Async metadata logging
- [x] Multi-modal content support
- [ ] Backend API
- [ ] Dashboard UI
- [ ] Closed beta (week 5)
- [ ] Public launch (week 6)

Follow [@bynimer](https://twitter.com/bynimer) for weekly progress updates, or join the waitlist at [nimer.dev](https://nimer.dev).

## Development

```bash
git clone https://github.com/nimer-dev/optimizer-sdk
cd optimizer-sdk
pip install -e ".[dev]"
pytest
```

The router is the most important piece — `tests/test_router.py` covers the routing rules.

## FAQ

**Is this actually a drop-in replacement?**
Yes. The `client.messages.create(...)` signature matches the Anthropic SDK. The only addition is `auto_route=True`, which defaults to on. Set it to `False` and pass `model=` to keep the original behavior.

**What if your routing picks the wrong model?**
Override on a per-call basis with `auto_route=False, model="..."`. The default is optimized for cost; you keep full control when you need it.

**How is this different from Helicone or Langfuse?**
Those are full observability platforms — they log everything and offer rich tracing. Nimer focuses on one thing: cost-aware routing for Claude. If you need full LLM observability, use those. If you want to cut your bill in half with three lines of code, use this.

**Will this work with streaming, tool use, or vision?**
Yes. The SDK forwards everything to the underlying Anthropic client unchanged.

**Why Claude only?**
Because narrow beats broad in v1. Once we're great at Claude, we'll consider OpenAI and Gemini.

**Where are you based?**
Saudi Arabia. Building globally.

## Founder

Built by [Majdi](https://twitter.com/trynimer) — a developer who turns problems into products. Started by watching his Claude API bill climb past $400/month and building a router to fix it. Named after his son Nimer, because the best things you build are for the people you love.

This is the first product. Not the last.

⭐ Star this repo if you've ever overspent on Claude API.

## License

MIT — see [LICENSE](LICENSE).
