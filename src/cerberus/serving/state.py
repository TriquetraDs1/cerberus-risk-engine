"""In-process serving state: a rolling per-account transaction window for velocity
features, and a graph-cache freshness flag for the graceful-degradation demo.

Explicitly a demo-scoped design, named as such: this state lives in one process's
memory and resets on restart. docs/ARCHITECTURE.md's Trade-off Analysis already names
the production path (a replicated store, e.g. Redis) — this is the honest, minimal
stand-in for it, not a production claim.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class AccountHistory:
    """Trailing transactions for one account, just enough to compute the same
    velocity and amount_zscore features the offline pipeline computes.
    """

    timestamps: list[datetime] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)

    def record(self, timestamp: datetime, amount: float) -> None:
        self.timestamps.append(timestamp)
        self.amounts.append(amount)

    def trailing(self, as_of: datetime, window: timedelta) -> tuple[int, float]:
        cutoff = as_of - window
        count, total = 0, 0.0
        for ts, amt in zip(self.timestamps, self.amounts, strict=True):
            if cutoff <= ts <= as_of:
                count += 1
                total += amt
        return count, total

    def mean_std(self) -> tuple[float, float]:
        if not self.amounts:
            return 0.0, 1.0
        import statistics

        mean = statistics.fmean(self.amounts)
        std = statistics.pstdev(self.amounts) if len(self.amounts) > 1 else 0.0
        return mean, std or 1.0


class ServingState:
    """Thread-safe holder for the process's in-memory serving state. One instance
    lives for the lifetime of the FastAPI app (see app.py's lifespan handler).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, AccountHistory] = defaultdict(AccountHistory)
        self.graph_status: str = "fresh"  # "fresh" | "degraded" — see /admin/graph-status

    def get_history(self, account_id: str) -> AccountHistory:
        """Returns the account's history object directly (mutable) — the caller
        should call `.record(...)` on it *before* reading trailing stats, matching
        the offline pipeline's convention that the window is inclusive of the
        transaction being scored (see features/pipeline.py's add_velocity_features).
        """
        with self._lock:
            return self._history[account_id]

    def set_graph_status(self, status: str) -> None:
        with self._lock:
            self.graph_status = status

    def is_graph_available(self) -> bool:
        return self.graph_status == "fresh"
