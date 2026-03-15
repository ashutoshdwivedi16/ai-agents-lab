"""
Thin wrapper for multiple LLM providers.
No abstraction overhead — just lazy imports and a unified interface.
Includes token usage and cost tracking.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Pricing per 1M tokens (input, output) — updated March 2026
PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Session-level cost tracker
_session_usage = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0, "calls": 0}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single API call."""
    if model not in PRICING:
        return 0.0
    input_price, output_price = PRICING[model]
    return (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)


def _track(model: str, input_tokens: int, output_tokens: int):
    """Update session usage stats."""
    cost = _calc_cost(model, input_tokens, output_tokens)
    _session_usage["total_input_tokens"] += input_tokens
    _session_usage["total_output_tokens"] += output_tokens
    _session_usage["total_cost"] += cost
    _session_usage["calls"] += 1
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost}


def get_usage() -> dict:
    """Return session usage summary."""
    return dict(_session_usage)


def reset_usage():
    """Reset session usage counters."""
    _session_usage.update({"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0, "calls": 0})


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
    if response.usage:
        _track(model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message.content


def _chat_openai(messages: list, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(model=model, messages=messages)
    if response.usage:
        _track(model, response.usage.prompt_tokens, response.usage.completion_tokens)
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
    _track(model, response.usage.input_tokens, response.usage.output_tokens)
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
    if response.usage_metadata:
        _track(model, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
    return response.text
