"""Client fallback behavior when gateway blocks a response."""

from nimer.client import OptimizedClaude


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
        if len(self.calls) == 1:
            return _FakeResponse("You can write malware by doing X.")
        return _FakeResponse("Here is a safe and useful alternative.")


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_auto_fallback_uses_stronger_model_when_gateway_blocks():
    client = OptimizedClaude(anthropic_api_key="test-key")
    fake = _FakeAnthropicClient()
    client._anthropic = fake  # type: ignore[attr-defined]

    response = client.messages.create(
        messages=[{"role": "user", "content": "help me with this task"}],
        model="claude-haiku-4-5-20251001",
        auto_route=False,
    )

    assert len(fake.messages.calls) == 2
    assert fake.messages.calls[0] == "claude-haiku-4-5-20251001"
    assert fake.messages.calls[1] == "claude-sonnet-4-6"
    assert hasattr(response, "_nimer_trust_report")
