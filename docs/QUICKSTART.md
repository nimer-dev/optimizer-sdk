# Nimer SDK Quickstart

## 1) Install

```bash
pip install nimer
```

## 2) Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export NIMER_API_KEY="nm_..."  # optional, enables dashboard analytics
```

## 3) First request

```python
from nimer import OptimizedClaude

client = OptimizedClaude()
resp = client.messages.create(
    max_tokens=256,
    messages=[{"role": "user", "content": "Translate good morning to Arabic."}],
)
print(resp.content[0].text)
```

## 4) Pick your integration mode

- Sync: `examples/basic_usage.py`
- Async: `examples/async_usage.py`
- Stream: `examples/stream_usage.py`
- Tools: `examples/tools_usage.py`

## 5) Verify savings in dashboard

When `NIMER_API_KEY` is set, each request logs usage metadata (tokens, selected model, estimated savings) to your dashboard.
