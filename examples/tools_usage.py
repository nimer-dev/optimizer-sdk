"""Tool-use example with Nimer SDK."""

from nimer import OptimizedClaude


def main() -> None:
    client = OptimizedClaude()

    response = client.messages.create(
        max_tokens=400,
        tools=[
            {
                "name": "get_weather",
                "description": "Fetches weather by city name",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What's the weather in Riyadh?"}],
    )

    print(response.content)


if __name__ == "__main__":
    main()
