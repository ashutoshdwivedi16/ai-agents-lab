"""Tests for chatbot helpers (no API calls needed)."""

from shared.utils.conversation import validate_input, trim_history


class TestValidateInput:
    def test_normal_input(self):
        assert validate_input("hello", 10000) is None

    def test_too_long(self):
        error = validate_input("x" * 10001, 10000)
        assert error is not None
        assert "too long" in error.lower()

    def test_exactly_at_limit(self):
        assert validate_input("x" * 10000, 10000) is None

    def test_empty_string(self):
        assert validate_input("", 10000) is None


class TestTrimHistory:
    def test_under_limit_unchanged(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = trim_history(messages, max_history=50)
        assert len(result) == 3

    def test_over_limit_preserves_system(self):
        system = {"role": "system", "content": "sys"}
        messages = [system] + [
            {"role": "user", "content": f"msg-{i}"} for i in range(60)
        ]
        result = trim_history(messages, max_history=10)
        assert result[0] == system
        assert len(result) == 11  # system + 10

    def test_keeps_most_recent(self):
        system = {"role": "system", "content": "sys"}
        messages = [system] + [
            {"role": "user", "content": f"msg-{i}"} for i in range(20)
        ]
        result = trim_history(messages, max_history=5)
        assert result[-1]["content"] == "msg-19"
        assert result[1]["content"] == "msg-15"

    def test_exactly_at_limit(self):
        system = {"role": "system", "content": "sys"}
        messages = [system] + [
            {"role": "user", "content": f"msg-{i}"} for i in range(50)
        ]
        result = trim_history(messages, max_history=50)
        assert len(result) == 51  # no trimming needed
