"""
Metrics subsystem — persistent, extensible, config-driven.

Public API:
    from shared.metrics import record_llm_call, get_metrics_summary, get_backend

Backend selected from config/default.yaml. Swap SQLite → Prometheus
by changing one YAML line — zero code changes.
"""

import uuid

from shared.metrics.backend import MetricsBackend
from shared.metrics.backends.noop_backend import NoopBackend
from shared.metrics.backends.sqlite_backend import SQLiteBackend
from shared.metrics.models import MetricRecord, MetricsSummary
from shared.logging import get_logger

logger = get_logger(__name__)

_backend: MetricsBackend | None = None
_session_id: str = ""


def init_metrics(config=None) -> None:
    """Initialize the metrics subsystem from config.

    Called once at startup. Safe to call multiple times (idempotent).
    """
    global _backend, _session_id

    if config is None:
        from shared.config import load_app_config
        app_config = load_app_config()
        config = app_config.metrics

    if not config.enabled:
        _backend = NoopBackend()
        logger.info("Metrics disabled")
        return

    _session_id = (
        str(uuid.uuid4())[:8] if config.session_id == "auto" else config.session_id
    )

    if config.backend == "sqlite":
        _backend = SQLiteBackend(config.sqlite_path)
    elif config.backend == "noop":
        _backend = NoopBackend()
    else:
        logger.warning(
            "Unknown metrics backend '%s', falling back to noop", config.backend
        )
        _backend = NoopBackend()

    logger.info("Metrics initialized: backend=%s session=%s", config.backend, _session_id)


def get_backend() -> MetricsBackend:
    """Return the active backend. Auto-initializes if needed."""
    global _backend
    if _backend is None:
        init_metrics()
    if _backend is None:
        raise RuntimeError("Metrics backend failed to initialize")
    return _backend


def get_session_id() -> str:
    """Return the current session ID."""
    if not _session_id:
        init_metrics()
    return _session_id


def record_llm_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    agent_name: str = "unknown",
    session_id: str | None = None,
) -> None:
    """Record metrics from a single LLM call. Fire-and-forget."""
    try:
        record = MetricRecord(
            session_id=session_id or get_session_id(),
            agent_name=agent_name,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        get_backend().record(record)
    except Exception:
        # Metrics must never crash the main application
        logger.exception("Failed to record metric")


def get_metrics_summary(**kwargs) -> MetricsSummary:
    """Return aggregated metrics summary."""
    return get_backend().summary(**kwargs)


def set_backend(backend: MetricsBackend) -> None:
    """Override the backend (for testing)."""
    global _backend
    _backend = backend


def set_session_id(sid: str) -> None:
    """Override the session ID (for testing)."""
    global _session_id
    _session_id = sid


def shutdown() -> None:
    """Clean up resources."""
    global _backend
    if _backend:
        _backend.close()
        _backend = None
