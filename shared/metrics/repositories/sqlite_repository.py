"""SQLite implementation of MetricsRepository."""

import sqlite3
from datetime import datetime
from pathlib import Path

from shared.metrics.repository import MetricsRepository
from shared.metrics.models import MetricRecord, MetricsSummary
from shared.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    agent_name      TEXT    NOT NULL DEFAULT 'unknown',
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost            REAL    NOT NULL DEFAULT 0.0,
    latency_ms      REAL    NOT NULL DEFAULT 0.0,
    timestamp       TEXT    NOT NULL
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_metrics_session   ON llm_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON llm_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_agent     ON llm_metrics(agent_name);
"""


class SQLiteRepository(MetricsRepository):
    """SQLite-backed metrics storage. Auto-creates tables on init.

    Supports context manager protocol for safe resource cleanup:
        with SQLiteRepository(":memory:") as repo:
            repo.save(record)
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("SQLiteRepository initialized: %s", db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(_CREATE_TABLE + _CREATE_INDEXES)
        self._conn.commit()

    def save(self, record: MetricRecord) -> None:
        self._conn.execute(
            """INSERT INTO llm_metrics
               (session_id, agent_name, provider, model,
                input_tokens, output_tokens, cost, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.session_id,
                record.agent_name,
                record.provider,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.cost,
                record.latency_ms,
                record.timestamp.isoformat(),
            ),
        )
        self._conn.commit()

    def query_summary(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> MetricsSummary:
        where_clauses, params = self._build_where(session_id, agent_name, since)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        row = self._conn.execute(
            f"""SELECT COUNT(*) as total_calls,
                       COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                       COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                       COALESCE(SUM(cost), 0.0) as total_cost,
                       COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
                FROM llm_metrics {where_sql}""",
            params,
        ).fetchone()

        by_provider = {}
        for r in self._conn.execute(
            f"SELECT provider, COUNT(*) as cnt FROM llm_metrics {where_sql} GROUP BY provider",
            params,
        ):
            by_provider[r["provider"]] = r["cnt"]

        by_model = {}
        for r in self._conn.execute(
            f"SELECT model, COUNT(*) as cnt FROM llm_metrics {where_sql} GROUP BY model",
            params,
        ):
            by_model[r["model"]] = r["cnt"]

        by_agent = {}
        for r in self._conn.execute(
            f"SELECT agent_name, COUNT(*) as cnt FROM llm_metrics {where_sql} GROUP BY agent_name",
            params,
        ):
            by_agent[r["agent_name"]] = r["cnt"]

        return MetricsSummary(
            total_calls=row["total_calls"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_cost=row["total_cost"],
            avg_latency_ms=row["avg_latency_ms"],
            by_provider=by_provider,
            by_model=by_model,
            by_agent=by_agent,
        )

    def query_records(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MetricRecord]:
        where_clauses, params = self._build_where(session_id, agent_name, since)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        rows = self._conn.execute(
            f"""SELECT session_id, agent_name, provider, model,
                       input_tokens, output_tokens, cost, latency_ms, timestamp
                FROM llm_metrics {where_sql}
                ORDER BY timestamp DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [
            MetricRecord(
                session_id=r["session_id"],
                agent_name=r["agent_name"],
                provider=r["provider"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cost=r["cost"],
                latency_ms=r["latency_ms"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _build_where(
        session_id: str | None,
        agent_name: str | None,
        since: datetime | None,
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if agent_name:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        return clauses, params
