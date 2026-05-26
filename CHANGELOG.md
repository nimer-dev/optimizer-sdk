# Changelog

All notable changes to `nimer` (the Python SDK) will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.1] — 2026-05-26

### Added
- `configure_otel_tracing()` — optional OTLP export helper for gateway traffic.
- `RetryPolicy` and typed errors (`RateLimitError`, `BudgetExceededError`, …) on HTTP paths.
- `nimer/_client_common.py` — shared sync/async core for `chat`, `ultrathink`, streaming, routing feedback.

### Changed
- `OptimizedClaude` / `AsyncNimer` refactored onto `_NimerBackendCore` (same public API).
- `_SecretStr` masks repr only so Bearer auth still works with `str` keys.

### Fixed
- Per-request HTTP timeouts and connection reuse hardened (enterprise audit follow-up).

## [0.2.0] — 2026-05-10

### Added
- **Multi-provider streaming** through Nimer's OpenAI-compatible
  `/v1/chat/completions` endpoint:
  - `OptimizedClaude.stream(messages, *, model=None, max_tokens=2048, **extra)`
    yields event dicts: `{"type": "delta" | "done" | "error", ...}`.
  - `OptimizedClaude.stream_text(...)` — `Iterator[str]` for the simple
    "just give me tokens" case.
  - `AsyncNimer.astream(...)` and `AsyncNimer.astream_text(...)` for async.
- `nimer/_streaming.py` — internal SSE parser + payload builder, isolated
  so `client.py` stays focused on the public surface.
- 13 new unit tests covering role/delta/done/error frames, the `[DONE]`
  terminator, comment / blank / wrong-prefix / malformed-JSON noise lines,
  payload builder invariants (forces `stream=True`, blocks `messages`
  override), and the no-key `ConfigurationError` path. **37/37 tests pass.**
- New examples: `examples/stream_usage.py` and `examples/astream_usage.py`.

### Changed
- HTTP errors during stream init now surface as `NimerError` with the API's
  `detail` message preserved verbatim (no opaque `httpx.HTTPStatusError`
  leak). For example, an invalid key produces:
  `Nimer streaming request failed (HTTP 401): Invalid API key`.
- `README.md` documents both streaming flows: the existing
  `messages.stream(...)` (Anthropic passthrough) and the new
  `client.stream(...)` (multi-provider, auto-routed).

### Internal
- `nimer/_version.py` bumped to `0.2.0`.
- Persistent `httpx.Client` reuse on `chat()` / `ultrathink()` paths
  (carried over from `4106e98` on `main`).

### Backwards compatibility
- `messages.create(...)`, `messages.stream(...)` (Anthropic passthrough),
  `chat()`, and `ultrathink()` are unchanged.
- Existing v0.1.x users see no breaking changes; `pip install -U nimer`
  unlocks the new streaming methods immediately.

## [0.1.0] — earlier

Initial release: rule-based router, Anthropic passthrough, Trust Gateway,
async metadata logging, and the multi-provider `chat()` / `ultrathink()`
entry points to the Nimer HTTP API.
