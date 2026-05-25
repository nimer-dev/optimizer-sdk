"""Rule-based routing logic.

This is the heart of the product. Start with simple, deterministic rules
that we can explain to users in a single sentence — then iterate based
on real usage data once we have it.

Design principles:
- Deterministic: same input -> same model, every time.
- Cheap: routing decision must take << 1ms (no API calls).
- Explainable: a user should be able to read the code and understand
  exactly why their request went to Haiku instead of Sonnet.
"""

from __future__ import annotations

from typing import Any, Iterable

from ._constants import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET


class Router:
    """Pick the cheapest Claude model that can handle a given request."""

    # Token thresholds (tiktoken when installed; script-aware heuristic otherwise).
    SHORT_INPUT_TOKENS = 50
    LONG_INPUT_TOKENS = 1250
    LONG_CODE_TOKENS = 125

    # Keywords that suggest code is in the request. Cheap heuristic; we'll
    # replace this with something smarter once we have usage data.
    CODE_MARKERS = ("```", "def ", "class ", "function ", "import ", "const ", "=> {")

    def choose(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
    ) -> str:
        """Select the model to use for this request."""
        last_user = self._last_user_text(messages)
        total_tokens = self._total_input_tokens(messages, system)
        last_user_tokens = self._estimate_tokens(last_user)
        has_code = self._looks_like_code(last_user)

        if total_tokens > self.LONG_INPUT_TOKENS:
            return MODEL_SONNET

        if has_code and last_user_tokens > self.LONG_CODE_TOKENS:
            return MODEL_SONNET

        if last_user_tokens < self.SHORT_INPUT_TOKENS:
            return MODEL_HAIKU

        # Everything in between: Sonnet is the safe middle ground.
        return MODEL_SONNET

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        """Return the text of the most recent user message, or ''."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            return _content_to_text(msg.get("content"))
        return ""

    @classmethod
    def _total_input_tokens(
        cls,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
    ) -> int:
        parts: list[str] = []
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            parts.extend(_content_to_text(part) for part in system)
        for msg in messages:
            parts.append(_content_to_text(msg.get("content")))
        return cls._estimate_tokens("".join(parts))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
        # Arabic/CJK: ~1 char ≈ 1 token; Latin-heavy English: ~4 chars/token.
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii > len(text) * 0.12:
            return len(text)
        return max(1, len(text) // 4)

    @classmethod
    def _looks_like_code(cls, text: str) -> bool:
        return any(marker in text for marker in cls.CODE_MARKERS)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic-style content into a plain string for sizing.

    The Anthropic API accepts strings, dicts (single content block), or
    lists of content blocks. We only care about text length here, so we
    walk the structure and concatenate any text we find.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _block_to_text(content)
    if isinstance(content, Iterable):
        return "".join(_block_to_text(part) for part in content)
    return ""


def _block_to_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "") or ""
    return ""
