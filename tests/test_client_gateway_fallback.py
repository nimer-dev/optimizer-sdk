"""Client must not escalate models when Trust Gateway blocks a response."""

import pytest

from nimer.client import OptimizedClaude
from nimer.exceptions import TrustGatewayError


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, input_tokens: int = 10, output_tokens: int = 15) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.calls.append(model)
        return _FakeResponse("You can write malware by doing X.")


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_gateway_block_does_not_escalate_to_expensive_model():
    client = OptimizedClaude(anthropic_api_key="test-key")
    fake = _FakeAnthropicClient()
    client._anthropic = fake  # type: ignore[attr-defined]

    with pytest.raises(TrustGatewayError):
        client.messages.create(
            messages=[{"role": "user", "content": "help me with this task"}],
            model="claude-haiku-4-5-20251001",
            auto_route=False,
        )

    assert fake.messages.calls == ["claude-haiku-4-5-20251001"]
