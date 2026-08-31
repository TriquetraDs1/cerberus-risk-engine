"""Turns a SHAP attribution row into human-readable reason codes — shared by the
offline dashboard export (scripts/export_dashboard_data.py) and the live /score
endpoint (cerberus.serving.app), so "why was this flagged" means the same thing in
both places rather than silently drifting apart.
"""

from __future__ import annotations

import numpy as np

from cerberus.features.pipeline import FEATURE_COLUMNS

# Human-readable labels for the point-risk model's features — this is what turns a
# SHAP value into the `reason_codes` field of the /score API contract in
# docs/ARCHITECTURE.md, rather than leaving feature names as internal jargon.
FEATURE_REASON_LABELS = {
    "amount": "large_transaction_amount",
    "amount_zscore": "amount_anomalous_for_account",
    "amount_vs_trailing_mean": "amount_far_above_account_norm",
    "velocity_count_1h": "high_transaction_velocity",
    "velocity_amount_1h": "high_value_velocity",
    "velocity_count_24h": "elevated_daily_transaction_count",
    "velocity_amount_24h": "elevated_daily_spend",
    "velocity_count_7d": "sustained_weekly_activity",
    "velocity_amount_7d": "sustained_weekly_spend",
    "hours_since_last_txn": "unusual_gap_since_last_transaction",
    "hour_of_day": "unusual_hour",
    # hour_sin / hour_cos are the cyclical encoding of the same clock reading, so they
    # share hour_of_day's label — an analyst should never see "hour cos" as a reason.
    "hour_sin": "unusual_hour",
    "hour_cos": "unusual_hour",
    "is_off_hours": "off_hours_transaction",
    "day_of_week": "unusual_day_pattern",
    "entity_degree": "linked_to_multiple_accounts",
    "shared_entity_strength": "strong_shared_identifier_links",
    "component_size": "part_of_large_linked_cluster",
}


def reason_codes_for_row(shap_row: np.ndarray, ring_id: str | None) -> list[str]:
    contributions = list(zip(FEATURE_COLUMNS, shap_row, strict=True))
    # only positive contributions (pushing toward fraud) are useful "reasons"; segment
    # one-hot columns are excluded — "segment_travel_luxury=1" isn't a human reason,
    # it's plumbing for the decision layer's threshold choice, not a risk signal to cite.
    positive = [(f, v) for f, v in contributions if v > 0 and not f.startswith("segment_")]
    positive.sort(key=lambda x: x[1], reverse=True)

    # Several features now map to the same human label (the three velocity windows, the
    # cyclical hour encodings). Dedupe by label, keeping the highest-attribution one, so
    # a flag never reads "unusual hour, unusual hour".
    reasons: list[str] = []
    for feature, _ in positive:
        label = FEATURE_REASON_LABELS.get(feature, feature)
        if label not in reasons:
            reasons.append(label)
        if len(reasons) == 2:
            break

    if ring_id:
        reasons.append(f"shared_device_with_flagged_ring:{ring_id}")
    return reasons or ["low_risk_no_dominant_factor"]
