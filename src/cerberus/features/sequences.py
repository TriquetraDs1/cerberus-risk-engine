"""Per-account transaction sequences for the sequence-risk model (roadmap B2).

The point-risk model sees one transaction at a time, with history compressed into
trailing-window aggregates. That compression is exactly what the slow-ramp evasion
exploits: stretch a coordinated burst past the widest window and the aggregates go
quiet, even though the *ordering* of the account's transactions is still anomalous. A
model that reads the sequence itself doesn't depend on any particular window width.

Each row becomes: the `SEQUENCE_LENGTH` transactions **ending at** this one, from the
same account, oldest-first — and the label of the final transaction. Strictly causal by
construction: a step never sees a transaction that comes after the one being scored, and
accounts with a short history are left-padded rather than dropped, so the model must
learn to work with partial context the way the live API will have to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cerberus.data.synthetic_rings import SEGMENTS

SEQUENCE_LENGTH = 8

# Per timestep: log amount, log inter-arrival gap, cyclical hour, one-hot segment.
# Deliberately raw-ish — the whole point is to let the model find temporal structure
# rather than to hand it the same engineered aggregates the point-risk model already has.
SEQUENCE_FEATURE_NAMES = ["log_amount", "log_gap_hours", "hour_sin", "hour_cos", *(f"segment_{s}" for s in SEGMENTS)]
N_SEQUENCE_FEATURES = len(SEQUENCE_FEATURE_NAMES)


def _per_step_matrix(txns: pd.DataFrame) -> np.ndarray:
    """One feature vector per transaction, in the order given."""
    ts = pd.to_datetime(txns["timestamp"])
    log_amount = np.log1p(txns["amount"].to_numpy(dtype=float))

    gap_hours = txns.groupby("account_id")["timestamp"].diff().dt.total_seconds() / 3600.0
    # A first transaction has no predecessor: treat it as a long quiet period, matching
    # the convention in features/pipeline.add_trailing_amount_features.
    log_gap = np.log1p(gap_hours.fillna(24.0 * 30).clip(lower=0).to_numpy(dtype=float))

    radians = 2 * np.pi * ts.dt.hour.to_numpy(dtype=float) / 24.0
    columns = [log_amount, log_gap, np.sin(radians), np.cos(radians)]
    columns.extend((txns["segment"] == seg).to_numpy(dtype=float) for seg in SEGMENTS)
    return np.column_stack(columns)


def build_sequences(
    txns: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return `(X, y, ordered_txns)`.

    `X` has shape `(n_transactions, sequence_length, N_SEQUENCE_FEATURES)`, `y` is the
    label of each window's final transaction, and `ordered_txns` is the transaction table
    in the exact row order of `X` — so a caller can apply the same chronological split
    boundaries the point-risk model uses and know the rows line up.
    """
    ordered = txns.sort_values("timestamp").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
    n = len(ordered)

    # Group each account's rows together while preserving time order inside the group.
    # With rows contiguous per account, "k transactions ago" becomes a plain array shift
    # instead of a per-account lookup — the difference between O(n) and O(accounts x n).
    by_account = ordered.sort_values(["account_id", "timestamp"], kind="stable")
    steps = _per_step_matrix(by_account)
    order_in_account = by_account.groupby("account_id").cumcount().to_numpy()

    grouped = np.zeros((n, sequence_length, N_SEQUENCE_FEATURES), dtype=np.float32)
    row_index = np.arange(n)
    for offset in range(sequence_length):
        # offset 0 is the transaction being scored; offset k is k steps earlier. Rows
        # whose account history is shorter than the offset keep their zero padding.
        valid = order_in_account >= offset
        grouped[valid, sequence_length - 1 - offset, :] = steps[row_index[valid] - offset]

    # Back to time order, so X lines up row-for-row with `ordered` and the same
    # chronological split boundaries apply.
    position_in_grouped = pd.Series(np.arange(n), index=by_account.index)
    X = grouped[position_in_grouped.reindex(ordered.index).to_numpy()]
    y = ordered["label"].to_numpy(dtype=np.float32)
    return X, y, ordered
