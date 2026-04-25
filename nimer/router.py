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

    # Tunable thresholds. These are chars, not tokens, because counting
    # tokens precisely requires a tokenizer call we want to avoid.
    # Rough rule of thumb: 1 token ≈ 4 chars for English.
    SHORT_INPUT_CHARS = 200       # below this, the task is almost certainly trivial
    LONG_INPUT_CHARS = 5000       # above this, we want a stronger model
    LONG_CODE_CHARS = 500         # code requests above this go to Opus

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
        total_chars = self._total_input_chars(messages, system)
        has_code = self._looks_like_code(last_user)

        # Long context — needs a model with strong long-context handling.
        if total_chars > self.LONG_INPUT_CHARS:
            return MODEL_SONNET

        # Substantial code-generation or code-understanding tasks.
        if has_code and len(last_user) > self.LONG_CODE_CHARS:
            return MODEL_OPUS

        # Short, simple prompts: classification, lookups, casual Q&A.
        if len(last_user) < self.SHORT_INPUT_CHARS:
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

    @staticmethod
    def _total_input_chars(
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
    ) -> int:
        """Approximate total prompt size in characters."""
        total = 0
        if isinstance(system, str):
            total += len(system)
        elif isinstance(system, list):
            total += sum(len(_content_to_text(part)) for part in system)

        for msg in messages:
            total += len(_content_to_text(msg.get("content")))
        return total

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
