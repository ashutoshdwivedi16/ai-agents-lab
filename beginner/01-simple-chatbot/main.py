"""
01 - Simple Chatbot Agent

A basic conversational agent that demonstrates:
- Setting up an LLM connection
- System prompts
- Conversation loop with chat history
- Graceful exit handling

Usage:
    python main.py
    python main.py --provider openai
    python main.py --provider anthropic --model claude-sonnet-4-5-20250929
"""

import sys
import os
import argparse

# Add project root to path so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.llm import chat, get_usage

SYSTEM_PROMPT = """You are a helpful, friendly assistant. Keep your responses concise \
and conversational. If you don't know something, say so honestly."""


def main():
    parser = argparse.ArgumentParser(description="Simple Chatbot Agent")
    parser.add_argument("--provider", default="groq", help="LLM provider (groq, openai, anthropic, gemini)")
    parser.add_argument("--model", default=None, help="Model name (uses provider default if not set)")
    args = parser.parse_args()

    print(f"Simple Chatbot | provider: {args.provider} | model: {args.model or 'default'}")
    print("Type 'quit' or 'exit' to stop.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat(messages, provider=args.provider, model=args.model)
            usage = get_usage()
            print(f"Bot: {response}")
            print(f"    [{usage['calls']} calls | {usage['total_input_tokens']+usage['total_output_tokens']} tokens | ${usage['total_cost']:.6f}]\n")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}\n")
            messages.pop()  # Remove failed user message from history


def _print_session_summary():
    usage = get_usage()
    if usage["calls"] == 0:
        return
    print(f"\n--- Session Summary ---")
    print(f"  API calls:     {usage['calls']}")
    print(f"  Input tokens:  {usage['total_input_tokens']}")
    print(f"  Output tokens: {usage['total_output_tokens']}")
    print(f"  Total cost:    ${usage['total_cost']:.6f}")
    print(f"-----------------------")


if __name__ == "__main__":
    main()
