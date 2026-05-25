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

_TIKTOKEN_ENC = None


def _get_tiktoken_enc():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _TIKTOKEN_ENC


class Router:
    """Pick the cheapest Claude model that can handle a given request."""

    CODE_MARKERS = ("```", "def ", "class ", "function ", "import ", "const ", "=> {")

    def __init__(
        self,
        *,
        short_input_tokens: int = 50,
        long_input_tokens: int = 1250,
        long_code_tokens: int = 125,
    ) -> None:
        self._short_input_tokens = short_input_tokens
        self._long_input_tokens = long_input_tokens
        self._long_code_tokens = long_code_tokens

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

        if total_tokens > self._long_input_tokens:
            return MODEL_SONNET

        if has_code and last_user_tokens > self._long_code_tokens:
            return MODEL_SONNET

        if last_user_tokens < self._short_input_tokens:
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
        enc = _get_tiktoken_enc()
        if enc is not None:
            return len(enc.encode(text))
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
