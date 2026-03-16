"""Session-level usage tracking with config-driven pricing."""

from shared.models import SessionUsageReport
from shared.config import load_app_config


class SessionUsage:
    """Tracks cumulative token usage and cost across a session."""

    def __init__(self):
        self._data = SessionUsageReport()
        self._config = load_app_config()

    def track(self, model: str, input_tokens: int, output_tokens: int):
        """Record usage from a single API call."""
        cost = self._calc_cost(model, input_tokens, output_tokens)
        self._data.total_input_tokens += input_tokens
        self._data.total_output_tokens += output_tokens
        self._data.total_cost += cost
        self._data.calls += 1

    def _calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost from config-driven pricing table."""
        for prov_config in self._config.providers.values():
            if model in prov_config.pricing:
                inp_price, out_price = prov_config.pricing[model]
                return (input_tokens * inp_price / 1_000_000) + (
                    output_tokens * out_price / 1_000_000
                )
        return 0.0

    def reset(self):
        """Reset all counters."""
        self._data = SessionUsageReport()

    def to_dict(self) -> dict:
        """Return usage as a plain dict."""
        return self._data.model_dump()
