"""
main.py
CLI runner for the AutoStream Conversational Agent.
Run with:  python main.py
"""

import os
import re
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv()

# Validate API key early
if not os.getenv("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY not set. Create a .env file or export the variable.")
    sys.exit(1)

from agent.graph import chat, initial_state

HEADER = """AutoStream AI Assistant
AI-powered video editing platform
---------------------------------"""

SEPARATOR = "-" * 65
WRAP_WIDTH = 78


def _strip_emoji(text: str) -> str:
    """Remove emoji and pictographic symbols for a clean terminal demo."""
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_re.sub("", text)


def _format_text(text: str, width: int = WRAP_WIDTH) -> str:
    """Wrap assistant text while preserving paragraphs and simple bullet lists."""
    cleaned = _strip_emoji(text).strip()
    if not cleaned:
        return ""

    blocks: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            blocks.append("")
            continue

        if line.startswith("* ") or line.startswith("- "):
            blocks.append(
                textwrap.fill(
                    line[2:].strip(),
                    width=width,
                    initial_indent="* ",
                    subsequent_indent="  ",
                )
            )
            continue

        blocks.append(textwrap.fill(line, width=width))

    return "\n".join(blocks)


def _print_message(label: str, text: str) -> None:
    print(f"{label}:")
    print(_format_text(text))
    print()


def run():
    print(HEADER)
    print("Type 'quit' or 'exit' to end the conversation.")
    print(SEPARATOR)
    print()

    state = initial_state()

    # Opening message from the agent
    opening = (
        "Welcome to AutoStream, your AI-powered video editing platform. "
        "I can help you with pricing, features, or getting started. What can I do for you today?"
    )
    _print_message("Assistant", opening)

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print()
            print(SEPARATOR)
            _print_message("Assistant", "Thank you for your time. Have a great day.")
            break

        try:
            state, reply = chat(state, user_input)
            print()
            print(SEPARATOR)
            _print_message("Assistant", reply)
        except Exception as e:
            print()
            print(SEPARATOR)
            _print_message("System", f"Error: {e}")
            _print_message("System", "Please check your GOOGLE_API_KEY and try again.")


if __name__ == "__main__":
    run()
