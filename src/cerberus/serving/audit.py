"""Append-only decision audit log, backed by SQLite — the `decisions` table from
docs/ARCHITECTURE.md's storage design. SQLite is the documented demo-scoped choice
(zero setup for a reviewer); Postgres is the noted production path.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    segment TEXT NOT NULL,
    amount REAL NOT NULL,
    risk_score REAL NOT NULL,
    decision TEXT NOT NULL,
    ring_id TEXT,
    ring_check TEXT NOT NULL,
    model_version TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    scored_at TEXT NOT NULL
);
"""


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        transaction_id: str,
        account_id: str,
        segment: str,
        amount: float,
        risk_score: float,
        decision: str,
        ring_id: str | None,
        ring_check: str,
        model_version: str,
        reason_codes: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO decisions
                (transaction_id, account_id, segment, amount, risk_score, decision,
                 ring_id, ring_check, model_version, reason_codes, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transaction_id,
                    account_id,
                    segment,
                    amount,
                    risk_score,
                    decision,
                    ring_id,
                    ring_check,
                    model_version,
                    ",".join(reason_codes),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get(self, transaction_id: str) -> dict | None:
        """Most recent decision row for a transaction id, or None. Used by /explain."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM decisions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
                (transaction_id,),
            ).fetchone()
            return dict(row) if row else None
