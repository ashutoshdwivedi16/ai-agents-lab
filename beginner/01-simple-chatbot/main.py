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

from shared.llm import chat

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
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat(messages, provider=args.provider, model=args.model)
            print(f"Bot: {response}\n")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}\n")
            messages.pop()  # Remove failed user message from history


if __name__ == "__main__":
    main()
