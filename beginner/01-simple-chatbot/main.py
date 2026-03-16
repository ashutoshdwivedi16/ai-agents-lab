"""
01 - Simple Chatbot Agent

A basic conversational agent that demonstrates:
- Config-driven system prompts (YAML, not hardcoded)
- Input validation and conversation bounds
- Structured logging
- Clean LLM provider abstraction

Usage:
    python main.py
    python main.py --provider openai
    python main.py --provider anthropic --model claude-sonnet-4-5-20250929
"""

import argparse

from shared.llm import chat, get_usage
from shared.config import load_agent_config
from shared.logging import get_logger
from shared.utils.conversation import validate_input, trim_history

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Simple Chatbot Agent")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (groq, openai, anthropic, gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (uses provider default if not set)",
    )
    args = parser.parse_args()

    config = load_agent_config("simple-chatbot")
    provider = args.provider or config.provider
    model = args.model or config.model

    logger.info("Starting chatbot: provider=%s model=%s", provider, model or "default")
    print(f"Simple Chatbot | provider: {provider} | model: {model or 'default'}")
    print("Type 'quit' or 'exit' to stop.\n")

    messages = [{"role": "system", "content": config.system_prompt}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            _print_session_summary()
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            _print_session_summary()
            print("Bye!")
            break

        error = validate_input(user_input, config.max_input_length)
        if error:
            print(f"  [{error}]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages, config.max_history)

        try:
            response = chat(messages, provider=provider, model=model)
            usage = get_usage()
            print(f"Bot: {response}")
            print(
                f"    [{usage['calls']} calls | "
                f"{usage['total_input_tokens']+usage['total_output_tokens']} tokens | "
                f"${usage['total_cost']:.6f}]\n"
            )
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            print(f"Error: {e}\n")
            messages.pop()  # Remove failed user message


def _print_session_summary():
    usage = get_usage()
    if usage["calls"] == 0:
        return
    print("\n--- Session Summary ---")
    print(f"  API calls:     {usage['calls']}")
    print(f"  Input tokens:  {usage['total_input_tokens']}")
    print(f"  Output tokens: {usage['total_output_tokens']}")
    print(f"  Total cost:    ${usage['total_cost']:.6f}")
    print("-----------------------")


if __name__ == "__main__":
    main()
