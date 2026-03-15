"""
Thin wrapper for multiple LLM providers.
No abstraction overhead — just lazy imports and a unified interface.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def chat(messages: list, provider: str = "groq", model: str = None) -> str:
    """Send messages to an LLM and return the response text."""

    if provider == "groq":
        return _chat_groq(messages, model or "llama-3.3-70b-versatile")
    elif provider == "openai":
        return _chat_openai(messages, model or "gpt-4o-mini")
    elif provider == "anthropic":
        return _chat_anthropic(messages, model or "claude-sonnet-4-5-20250929")
    elif provider == "gemini":
        return _chat_gemini(messages, model or "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _chat_groq(messages: list, model: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def _chat_openai(messages: list, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def _chat_anthropic(messages: list, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Anthropic uses system param separately
    system = None
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_messages.append(msg)

    kwargs = {"model": model, "max_tokens": 1024, "messages": chat_messages}
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _chat_gemini(messages: list, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if msg["role"] == "system":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        else:
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    response = client.models.generate_content(model=model, contents=contents)
    return response.text
