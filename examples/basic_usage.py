"""Basic usage example for the Nimer Optimizer SDK.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    export NIMER_API_KEY=nm_...        # optional
    python examples/basic_usage.py
"""

from nimer import OptimizedClaude


def main() -> None:
    # Picks up ANTHROPIC_API_KEY and NIMER_API_KEY from the environment.
    client = OptimizedClaude()

    # 1. Auto-routed call — SDK picks the cheapest model that fits.
    short_response = client.messages.create(
        max_tokens=128,
        messages=[
            {"role": "user", "content": "Translate 'good morning' to Arabic."}
        ],
    )
    print("[auto] short prompt ->", _first_text(short_response))

    # 2. Manual override — route explicitly when you want full control.
    forced_response = client.messages.create(
        max_tokens=256,
        model="claude-sonnet-4-6",
        auto_route=False,
        messages=[
            {"role": "user", "content": "Summarize the theory of relativity in 3 sentences."}
        ],
    )
    print("[manual] sonnet ->", _first_text(forced_response))


def _first_text(response) -> str:
    """Pull the first text block out of an Anthropic response."""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


if __name__ == "__main__":
    main()
