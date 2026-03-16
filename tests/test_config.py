"""Tests for config loading."""

from pathlib import Path

from shared.config import load_app_config, load_agent_config


class TestAppConfig:
    def test_loads_default_yaml(self):
        config = load_app_config()
        assert config.default_provider == "groq"
        assert "groq" in config.providers
        assert "openai" in config.providers
        assert config.max_retries == 3

    def test_groq_pricing_loaded(self):
        config = load_app_config()
        groq = config.providers["groq"]
        assert "llama-3.3-70b-versatile" in groq.pricing
        assert groq.env_key == "GROQ_API_KEY"

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_app_config(tmp_path / "nonexistent.yaml")
        assert config.default_provider == "groq"
        assert config.providers == {}


class TestAgentConfig:
    def test_loads_chatbot_config(self):
        config = load_agent_config("simple-chatbot")
        assert "helpful" in config.system_prompt.lower()
        assert config.max_history == 50
        assert config.provider == "groq"

    def test_missing_agent_returns_defaults(self):
        config = load_agent_config("nonexistent-agent")
        assert config.provider == "groq"
        assert config.max_history == 50
