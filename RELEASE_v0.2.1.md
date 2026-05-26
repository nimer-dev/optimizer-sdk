# Release v0.2.1

## Install

```bash
pip install -U "nimer==0.2.1"
```

## Highlights

- Shared `_NimerBackendCore` for sync/async chat, ultrathink, streaming, routing feedback
- `configure_otel_tracing()` for optional OTLP export
- `RetryPolicy` and typed HTTP errors (`RateLimitError`, `BudgetExceededError`, …)
- Enterprise audit hardening: HTTP timeouts and connection reuse

## Publish checklist (founder)

1. Repo secret `PYPI_API_TOKEN` on `nimer-dev/optimizer-sdk`
2. GitHub → Releases → tag `v0.2.1` → paste this file as notes → Publish
3. Verify workflow **Publish to PyPI** is green
4. `pip install -U nimer==0.2.1`
