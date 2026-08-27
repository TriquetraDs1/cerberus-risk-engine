"""Feature engineering for the point-risk model.

Velocity, amount z-score, and time-of-day come straight from the transaction table.
`entity_degree` is a light preview of the Day 3 graph layer: how many other accounts an
account's device/ip/card is linked to. It's deliberately cheap (a lookup, not a full
community-detection pass) so the point-risk model can use "this account's device is
linked to N other accounts" as a signal today, before the Louvain ring detector exists.
"""

from __future__ import annotations

import pandas as pd


def add_velocity_features(txns: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """Trailing transaction count and amount sum per account within `window`
    (inclusive of the transaction itself, as of its own timestamp).
    """
    txns = txns.sort_values("timestamp").copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    count_col = f"velocity_count_{window}"
    sum_col = f"velocity_amount_{window}"
    txns[count_col] = 0.0
    txns[sum_col] = 0.0

    for _, group in txns.groupby("account_id"):
        amt_by_time = pd.Series(group["amount"].values, index=pd.DatetimeIndex(group["timestamp"].values))
        txns.loc[group.index, count_col] = amt_by_time.rolling(window).count().values
        txns.loc[group.index, sum_col] = amt_by_time.rolling(window).sum().values

    return txns


def add_amount_zscore(txns: pd.DataFrame) -> pd.DataFrame:
    """Per-account amount z-score against that account's own history (falls back to the
    global distribution for accounts with a single transaction).
    """
    txns = txns.copy()
    global_mean, global_std = txns["amount"].mean(), txns["amount"].std() + 1e-9

    stats = txns.groupby("account_id")["amount"].agg(["mean", "std"])
    stats["std"] = stats["std"].fillna(global_std).replace(0, global_std)
    joined = txns.join(stats, on="account_id", rsuffix="_acct")
    txns["amount_zscore"] = (txns["amount"] - joined["mean"]) / joined["std"]
    txns["amount_zscore"] = txns["amount_zscore"].fillna(
        (txns["amount"] - global_mean) / global_std
    )
    return txns


def add_time_features(txns: pd.DataFrame) -> pd.DataFrame:
    txns = txns.copy()
    ts = pd.to_datetime(txns["timestamp"])
    txns["hour_of_day"] = ts.dt.hour
    txns["is_off_hours"] = txns["hour_of_day"].between(0, 5).astype(int)
    txns["day_of_week"] = ts.dt.dayofweek
    return txns


def add_entity_degree(txns: pd.DataFrame, entity_edges: pd.DataFrame) -> pd.DataFrame:
    """How many distinct accounts is this account linked to via any shared entity?
    A cheap preview of graph structure ahead of the Day 3 Louvain layer.
    """
    txns = txns.copy()
    if entity_edges.empty:
        txns["entity_degree"] = 0
        return txns

    degree = pd.concat([entity_edges["entity_a"], entity_edges["entity_b"]]).value_counts()
    txns["entity_degree"] = txns["account_id"].map(degree).fillna(0).astype(int)
    return txns


FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "velocity_count_1h",
    "velocity_amount_1h",
    "hour_of_day",
    "is_off_hours",
    "day_of_week",
    "entity_degree",
]


def build_features(txns: pd.DataFrame, entity_edges: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and return the transaction table with all
    FEATURE_COLUMNS populated, ready for `detection.point_risk.train`.
    """
    out = add_velocity_features(txns)
    out = add_amount_zscore(out)
    out = add_time_features(out)
    out = add_entity_degree(out, entity_edges)
    return out
