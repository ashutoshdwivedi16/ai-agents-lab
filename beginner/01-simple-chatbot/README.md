# 01 - Simple Chatbot

A basic conversational agent that demonstrates core concepts of building AI agents with production patterns.

## What You'll Learn

- Config-driven system prompts (YAML, not hardcoded)
- Input validation and conversation bounds
- Structured logging
- Switching between LLM providers (Groq, OpenAI, Anthropic, Gemini)
- Token usage and cost tracking

## Setup

```bash
# From the project root
cd ai-agents-lab

# Create virtual environment and install deps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set up your API key
cp .env.example .env
# Edit .env and add your Groq key (free tier)
```

## Usage

```bash
# Default: uses Groq (free tier) — run from project root
PYTHONPATH=. python beginner/01-simple-chatbot/main.py

# Switch provider
PYTHONPATH=. python beginner/01-simple-chatbot/main.py --provider openai

# Use a specific model
PYTHONPATH=. python beginner/01-simple-chatbot/main.py --provider groq --model llama-3.1-8b-instant
```

## How It Works

1. Loads system prompt from `config/agents/simple-chatbot.yaml`
2. Validates user input (rejects oversized messages)
3. Trims conversation history to prevent unbounded growth
4. Sends messages to the LLM, tracks token usage and cost
5. Shows running cost after each response + session summary on exit

## Config

Edit `config/agents/simple-chatbot.yaml` to change:
- `system_prompt` — the bot's personality
- `max_history` — max messages to keep (default: 50)
- `max_input_length` — max chars per input (default: 10000)
- `provider` — default provider (default: groq)
