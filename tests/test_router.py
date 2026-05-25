"""Tests for the routing logic.

We don't make any real API calls here — the router is pure logic.
"""

from nimer import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, Router


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def test_short_simple_prompt_routes_to_haiku():
    router = Router()
    chosen = router.choose([_user("What's the capital of France?")])
    assert chosen == MODEL_HAIKU


def test_medium_prompt_routes_to_sonnet():
    router = Router()
    medium = "Explain the difference between supervised and unsupervised learning. " * 5
    chosen = router.choose([_user(medium)])
    assert chosen == MODEL_SONNET


def test_long_context_routes_to_sonnet():
    router = Router()
    long_text = "context " * 1000  # ~8000 chars
    chosen = router.choose([_user(long_text)])
    assert chosen == MODEL_SONNET


def test_substantial_code_request_routes_to_sonnet():
    router = Router()
    code_request = (
        "Refactor this Python module to use async/await throughout. "
        + "```python\n"
        + "def process_data(items):\n    return [transform(i) for i in items]\n" * 30
        + "```\n"
        + "Also explain the tradeoffs of each change you make."
    )
    chosen = router.choose([_user(code_request)])
    assert chosen == MODEL_SONNET


def test_short_code_question_stays_on_haiku():
    """A tiny code question shouldn't trigger Opus."""
    router = Router()
    chosen = router.choose([_user("What does `def` mean in Python?")])
    assert chosen == MODEL_HAIKU


def test_system_prompt_counted_toward_total_size():
    router = Router()
    huge_system = "system instruction. " * 500  # ~10000 chars
    chosen = router.choose(
        [_user("hi")],
        system=huge_system,
    )
    assert chosen == MODEL_SONNET


def test_handles_multimodal_content_blocks():
    """Multi-modal messages use list-of-blocks format; we only size text."""
    router = Router()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this picture?"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
    ]
    chosen = router.choose(messages)
    assert chosen == MODEL_HAIKU


def test_uses_last_user_message_for_size_check():
    """When deciding short-vs-not, only the latest user turn matters."""
    router = Router()
    messages = [
        _user("a much longer earlier turn " * 30),
        {"role": "assistant", "content": "ok"},
        _user("short"),
    ]
    chosen = router.choose(messages)
    # Last user message is short, but total size is still under threshold
    # so we route to Haiku.
    assert chosen == MODEL_HAIKU


def test_empty_messages_does_not_crash():
    router = Router()
    chosen = router.choose([])
    # Empty input -> last_user is "" (length 0) -> Haiku branch
    assert chosen == MODEL_HAIKU
