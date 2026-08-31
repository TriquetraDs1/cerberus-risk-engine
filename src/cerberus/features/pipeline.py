"""Feature engineering for the point-risk model.

Velocity, amount z-score, and time-of-day come straight from the transaction table.
`entity_degree` is a light preview of the Day 3 graph layer: how many other accounts an
account's device/ip/card is linked to. It's deliberately cheap (a lookup, not a full
community-detection pass) so the point-risk model can use "this account's device is
linked to N other accounts" as a signal today, before the Louvain ring detector exists.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

# Fixed, known set — one-hot columns must be deterministic across train/calibration/
# test splits and across the export script's later re-featurization, so this list
# can't just be "whatever segments happen to appear in this dataframe."
from cerberus.data.synthetic_rings import SEGMENTS


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


# Multiple trailing windows, not just 1h. A single short window is precisely what the
# slow-ramp evasion strategy exploits: stretch a coordinated burst past the window and
# the velocity signal vanishes (see adversarial/strategies.py, apply_slow_ramp). Wider
# windows keep a stretched burst visible.
VELOCITY_WINDOWS = ("1h", "24h", "7d")


def add_multi_window_velocity(txns: pd.DataFrame) -> pd.DataFrame:
    out = txns
    for window in VELOCITY_WINDOWS:
        out = add_velocity_features(out, window=window)
    return out


def add_trailing_amount_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Amount relative to the account's own *trailing* history, plus the gap since its
    previous transaction.

    Deliberately expanding/shifted, not whole-dataset aggregates: at scoring time only
    the past exists, so a feature built from an account's full-history mean would be
    information the live API can't reproduce. `shift(1)` excludes the current row.
    """
    txns = txns.sort_values("timestamp").copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    grouped = txns.groupby("account_id")

    trailing_mean = grouped["amount"].transform(lambda s: s.shift(1).expanding().mean())
    # First transaction for an account has no history — fall back to its own amount, so
    # the ratio is a neutral 1.0 rather than NaN or a spurious spike.
    txns["amount_vs_trailing_mean"] = txns["amount"] / trailing_mean.fillna(txns["amount"]).replace(0, 1e-9)

    gap = grouped["timestamp"].diff().dt.total_seconds() / 3600.0
    # No previous transaction => treat as a long quiet period, not zero (zero would read
    # as "instantaneous repeat", the opposite of the truth).
    txns["hours_since_last_txn"] = gap.fillna(24.0 * 30)
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
    # Cyclical encoding: as a raw integer, hour 23 and hour 0 are maximally far apart,
    # so a tree has to spend splits rediscovering that midnight wraps. sin/cos puts them
    # adjacent. Kept alongside hour_of_day rather than replacing it — the raw integer is
    # still the readable one in a SHAP reason code.
    radians = 2 * np.pi * txns["hour_of_day"] / 24.0
    txns["hour_sin"] = np.sin(radians)
    txns["hour_cos"] = np.cos(radians)
    return txns


def add_segment_features(txns: pd.DataFrame) -> pd.DataFrame:
    """One-hot the merchant segment. Fraud economics genuinely differ by category
    (see cerberus.data.synthetic_rings.SEGMENT_PROFILES), so this is a real signal for
    the point-risk model, not just plumbing for the Day 4 decision layer — though the
    decision layer is what actually turns "which segment" into "which threshold."
    """
    txns = txns.copy()
    for seg in SEGMENTS:
        txns[f"segment_{seg}"] = (txns["segment"] == seg).astype(int)
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


def add_graph_features(txns: pd.DataFrame, entity_edges: pd.DataFrame) -> pd.DataFrame:
    """Two structural signals beyond raw degree.

    `shared_entity_strength` — the total edge weight on an account, where weight is the
    number of distinct shared identifiers (device AND card is a stronger link than device
    alone; see build_entity_edges). Degree counts *how many* neighbours, this counts *how
    tightly*.

    `component_size` — how many accounts sit in this account's connected component.

    **Neither is in FEATURE_COLUMNS, and that is a measured decision, not an oversight.**
    The original intent was exactly the opposite: give the point-risk model graph shape so
    it could react even when the ring detector stays silent. Running the adversarial
    harness with them included showed the reverse — identity-rotation evasion went from
    0.33 to 0.00 (total evasion), because the classifier learned to lean on structure that
    the rotation attack exists to destroy. Removing them restored the point-risk model's
    independent signal. See docs/EXPERIMENT_ADVANCED_TRAINING.md.

    They are computed here anyway: the ring detector and the dashboard both use them, and
    they are the obvious inputs for a future GNN (roadmap B1) whose whole job is graph
    structure. The lesson is about which *model* consumes them, not about the features.
    """
    txns = txns.copy()
    if entity_edges.empty:
        txns["shared_entity_strength"] = 0.0
        txns["component_size"] = 1
        return txns

    weights = pd.concat(
        [
            entity_edges[["entity_a", "weight"]].rename(columns={"entity_a": "account_id"}),
            entity_edges[["entity_b", "weight"]].rename(columns={"entity_b": "account_id"}),
        ]
    )
    strength = weights.groupby("account_id")["weight"].sum()
    txns["shared_entity_strength"] = txns["account_id"].map(strength).fillna(0.0).astype(float)

    graph = nx.from_pandas_edgelist(entity_edges, "entity_a", "entity_b")
    component_of = {
        node: len(component) for component in nx.connected_components(graph) for node in component
    }
    # An account with no edges is its own component of one.
    txns["component_size"] = txns["account_id"].map(component_of).fillna(1).astype(int)
    return txns


FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    *(f"velocity_count_{w}" for w in VELOCITY_WINDOWS),
    *(f"velocity_amount_{w}" for w in VELOCITY_WINDOWS),
    "amount_vs_trailing_mean",
    "hours_since_last_txn",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "is_off_hours",
    "day_of_week",
    "entity_degree",
    # NOTE: shared_entity_strength and component_size are deliberately NOT here.
    # See the ablation note in add_graph_features — feeding them to the point-risk
    # model made identity-rotation evasion strictly worse.
    *(f"segment_{seg}" for seg in SEGMENTS),
]


def build_features(txns: pd.DataFrame, entity_edges: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and return the transaction table with all
    FEATURE_COLUMNS populated, ready for `detection.point_risk.train`.
    """
    out = add_multi_window_velocity(txns)
    out = add_trailing_amount_features(out)
    out = add_amount_zscore(out)
    out = add_time_features(out)
    out = add_entity_degree(out, entity_edges)
    out = add_graph_features(out, entity_edges)
    out = add_segment_features(out)
    return out
