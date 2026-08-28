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
    "velocity_count_1h": "high_transaction_velocity",
    "velocity_amount_1h": "high_value_velocity",
    "hour_of_day": "unusual_hour",
    "is_off_hours": "off_hours_transaction",
    "day_of_week": "unusual_day_pattern",
    "entity_degree": "linked_to_multiple_accounts",
}


def reason_codes_for_row(shap_row: np.ndarray, ring_id: str | None) -> list[str]:
    contributions = list(zip(FEATURE_COLUMNS, shap_row, strict=True))
    # only positive contributions (pushing toward fraud) are useful "reasons"; segment
    # one-hot columns are excluded — "segment_travel_luxury=1" isn't a human reason,
    # it's plumbing for the decision layer's threshold choice, not a risk signal to cite.
    positive = [(f, v) for f, v in contributions if v > 0 and not f.startswith("segment_")]
    positive.sort(key=lambda x: x[1], reverse=True)
    reasons = [FEATURE_REASON_LABELS.get(f, f) for f, _ in positive[:2]]
    if ring_id:
        reasons.append(f"shared_device_with_flagged_ring:{ring_id}")
    return reasons or ["low_risk_no_dominant_factor"]
