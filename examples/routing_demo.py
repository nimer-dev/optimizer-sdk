"""Routing demo — see Nimer Optimizer pick models on real prompts.

This script makes 5 actual Anthropic API calls of varying complexity,
prints which model the router chose for each, and reports total
estimated savings vs. running everything on Sonnet (the naive default).

Setup:
    PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
    Then:        python examples/routing_demo.py

Cost: roughly $0.05-0.10 for a full run.
"""

from __future__ import annotations

import os
import sys
import textwrap

from nimer import BASELINE_MODEL, OptimizedClaude, estimate_cost


# Five prompts that exercise every branch of the router.
PROMPTS = [
    {
        "label": "1. Short factual",
        "expected": "Haiku (short, simple)",
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
    },
    {
        "label": "2. Short translation",
        "expected": "Haiku (short, simple)",
        "messages": [
            {
                "role": "user",
                "content": "Translate 'good morning, how are you?' to Arabic.",
            },
        ],
    },
    {
        "label": "3. Medium reasoning",
        "expected": "Sonnet (medium, no code)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain in 3-4 sentences why mutual TLS is more secure than "
                    "basic auth for service-to-service communication in a "
                    "microservices architecture. Mention the role of the "
                    "certificate authority and the trade-offs of managing "
                    "certificate rotation at scale. Be specific about which "
                    "attack vectors are mitigated."
                ),
            },
        ],
    },
    {
        "label": "4. Long context summary",
        "expected": "Sonnet (long input)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the following in 2 sentences:\n\n"
                    + (
                        "The history of computing began with the abacus and "
                        "progressed through mechanical calculators, vacuum tube "
                        "computers, transistors, integrated circuits, and finally "
                        "microprocessors that power modern devices. "
                    )
                    * 40
                ),
            },
        ],
    },
    {
        "label": "5. Substantial code task",
        "expected": "Opus (long + code)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Refactor this Python class to use async/await throughout, "
                    "add proper error handling with custom exception classes, "
                    "and explain each change you make. Include type hints.\n\n"
                    "```python\n"
                    + (
                        "class DataProcessor:\n"
                        "    def __init__(self, source):\n"
                        "        self.source = source\n"
                        "    def fetch(self, item_id):\n"
                        "        response = requests.get(f'{self.source}/{item_id}')\n"
                        "        return response.json()\n"
                        "    def process_batch(self, ids):\n"
                        "        return [self.fetch(i) for i in ids]\n"
                    )
                    * 6
                    + "```\n\n"
                    "Be thorough — discuss connection pooling, retries, and "
                    "rate limiting where relevant."
                ),
            },
        ],
    },
]


def first_text(response) -> str:
    """Pull the first text block from an Anthropic response."""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def short_model_name(full_name: str) -> str:
    """Turn 'claude-haiku-4-5-20251001' into 'Haiku'."""
    parts = full_name.split("-")
    if len(parts) >= 2:
        return parts[1].capitalize()
    return full_name


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print()
        print("In PowerShell, run:")
        print('    $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        return 1

    client = OptimizedClaude()

    total_actual = 0.0
    total_baseline = 0.0

    print("=" * 72)
    print("  Nimer Optimizer — Routing Demo")
    print("=" * 72)
    print()

    for case in PROMPTS:
        print(f">> {case['label']}")
        print(f"   expected: {case['expected']}")

        try:
            response = client.messages.create(
                max_tokens=400,
                messages=case["messages"],
            )
        except Exception as exc:
            print(f"   ERROR: {exc}")
            print()
            continue

        chosen = response.model
        usage = response.usage
        in_tok = usage.input_tokens
        out_tok = usage.output_tokens

        actual = estimate_cost(chosen, in_tok, out_tok)
        baseline = estimate_cost(BASELINE_MODEL, in_tok, out_tok)
        total_actual += actual
        total_baseline += baseline

        snippet = textwrap.shorten(
            first_text(response).strip().replace("\n", " "),
            width=180,
            placeholder="...",
        )

        print(f"   routed:   {short_model_name(chosen):<7} ({chosen})")
        print(f"   tokens:   {in_tok:>5} in  /  {out_tok:>4} out")
        print(f"   cost:     ${actual:.6f}   (baseline Sonnet: ${baseline:.6f})")
        print(f"   response: {snippet}")
        print()

    # Summary
    savings = total_baseline - total_actual
    pct = (savings / total_baseline * 100) if total_baseline > 0 else 0.0

    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Actual total cost:        ${total_actual:.6f}")
    print(f"  Baseline (all Sonnet):    ${total_baseline:.6f}")
    print(f"  Savings:                  ${savings:.6f}   ({pct:+.1f}%)")
    print()

    if pct > 0:
        print(f"  -> Saved {pct:.1f}% on this batch by routing to cheaper models.")
    elif pct < 0:
        print(f"  -> Spent {-pct:.1f}% more — routed up to Opus on heavy code task.")
        print("     This is intentional: cheap models would fail those prompts.")
    else:
        print("  -> Same cost as baseline.")

    print()
    print("  Note: this is one batch of 5 prompts. Real workloads with mostly")
    print("  short/medium tasks typically save 50-70% on the monthly bill.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
