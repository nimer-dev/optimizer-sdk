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

Built by [Nimer](https://twitter.com/trynimer) — someone who's been taking apart computers and putting them back together since childhood. Started with hardware: building rigs, swapping GPUs, pushing machines to their limits. Moved into software the same way — by actually doing it, including writing low-level code to unlock encrypted satellite streams just to watch a football match.

That same instinct — find the inefficiency, engineer around it — led to Nimer. Watched a Claude API bill climb past $400/month, understood exactly why, and built a router to fix it. Now turning it into a product, in public.

⭐ Star this repo if you've ever overspent on Claude API.

## License

MIT — see [LICENSE](LICENSE).
