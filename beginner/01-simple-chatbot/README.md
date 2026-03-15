# 01 - Simple Chatbot

A basic conversational agent that demonstrates core concepts of building AI agents.

## What You'll Learn

- Setting up an LLM API connection
- System prompts and how they shape agent behavior
- Building a conversation loop with chat history
- Switching between LLM providers (Groq, OpenAI, Anthropic, Gemini)

## Setup

```bash
cd beginner/01-simple-chatbot
pip install -r requirements.txt
cp .env.example .env
# Add your API key(s) to .env
```

## Usage

```bash
# Default: uses Groq (free tier)
python main.py

# Switch provider
python main.py --provider openai
python main.py --provider anthropic
python main.py --provider gemini

# Use a specific model
python main.py --provider groq --model llama-3.1-8b-instant
```

## How It Works

1. Loads a system prompt that defines the bot's personality
2. Takes user input in a loop
3. Sends the full conversation history to the LLM on each turn
4. Appends the response to history, so the bot remembers context
