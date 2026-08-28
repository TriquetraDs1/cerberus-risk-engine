#!/usr/bin/env python
"""Export real pipeline output as static JSON for the Next.js dashboard.

No mock data: every number here comes from the actual trained model (reloaded from
disk), the actual fitted calibrator, actual SHAP attributions, the actual Day 3 Louvain
output, and the actual Day 4 per-segment decision layer. Run after generate_data.py,
detect_rings.py, train_baseline.py, and build_decision_layer.py.

Usage:
    python scripts/generate_data.py
    python scripts/detect_rings.py
    python scripts/train_baseline.py
    python scripts/build_decision_layer.py
    python scripts/export_dashboard_data.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from cerberus.common.config import (
    BASELINE_METRICS_JSON,
    BASELINE_MODEL_PATH,
    CALIBRATOR_PATH,
    DASHBOARD_DATA_DIR,
    DETECTED_RINGS_JSON,
    REPORTS_DIR,
    RING_DETECTION_REPORT_JSON,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.detection.point_risk import three_way_split
from cerberus.features.pipeline import FEATURE_COLUMNS, build_features

DECISION_LAYER_JSON = REPORTS_DIR / "decision_layer.json"

QUEUE_SAMPLE_SIZE = 250

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


def load_detected_ring_membership() -> dict[str, str]:
    """account_id -> detected ring id, from the Day 3 Louvain output."""
    if not DETECTED_RINGS_JSON.exists():
        return {}
    detected = json.loads(DETECTED_RINGS_JSON.read_text())
    return {account: ring_id for ring_id, members in detected.items() for account in members}


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


def main() -> None:
    required = (SYNTHETIC_TRANSACTIONS_CSV, BASELINE_MODEL_PATH, CALIBRATOR_PATH, BASELINE_METRICS_JSON, DECISION_LAYER_JSON)
    for path in required:
        if not path.exists():
            raise SystemExit(
                f"Missing {path} — run generate_data.py, detect_rings.py, train_baseline.py, "
                "and build_decision_layer.py first."
            )

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading transactions and rebuilding features...")
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    features = build_features(txns, edges)
    # Same three-way split used at training time, so "the held-out set" here is
    # exactly the rows the model/calibrator/decision layer were evaluated against —
    # not a different, inconsistent slice.
    _, _, test_df = three_way_split(features)

    print("Loading trained booster + calibrator, scoring the held-out set...")
    booster = lgb.Booster(model_file=str(BASELINE_MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)
    X_test = test_df[FEATURE_COLUMNS]
    raw_scores = booster.predict(X_test)
    scores = calibrator.predict(raw_scores)  # calibrated P(fraud), what the product serves

    print("Computing SHAP attributions for reason codes...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):  # some shap/lightgbm version combos return [neg, pos]
        shap_values = shap_values[1]

    metrics = json.loads(BASELINE_METRICS_JSON.read_text())
    decision_layer = json.loads(DECISION_LAYER_JSON.read_text())
    segment_routing = decision_layer["segments"]
    ring_report = (
        json.loads(RING_DETECTION_REPORT_JSON.read_text())
        if RING_DETECTION_REPORT_JSON.exists()
        else None
    )
    rings_ground_truth = (
        json.loads(SYNTHETIC_RINGS_JSON.read_text()) if SYNTHETIC_RINGS_JSON.exists() else {}
    )
    ring_membership = load_detected_ring_membership()

    test_df = test_df.reset_index(drop=True)
    rows = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        score = float(scores[i])
        account_id = row["account_id"]
        segment = row["segment"]
        ring_id = ring_membership.get(account_id)

        seg_routing = segment_routing.get(segment)
        block_threshold = seg_routing["block_threshold"] if seg_routing else decision_layer["global_default_threshold"]
        review_threshold = seg_routing["review_threshold"] if seg_routing else block_threshold * 0.5
        decision = "block" if score >= block_threshold else "review" if score >= review_threshold else "approve"

        rows.append(
            {
                "transaction_id": row["transaction_id"],
                "account_id": account_id,
                "segment": segment,
                "timestamp": row["timestamp"].isoformat(),
                "amount": round(float(row["amount"]), 2),
                "risk_score": round(score, 4),
                "decision": decision,
                "reason_codes": reason_codes_for_row(shap_values[i], ring_id),
                "ring_id": ring_id,
                "actual_label": int(row["label"]),
                "cost_basis": {
                    "fp_cost": round(seg_routing["cost_matrix"]["fp_cost"], 2) if seg_routing else metrics["fp_cost"],
                    "fn_cost": round(seg_routing["cost_matrix"]["fn_cost"], 2) if seg_routing else metrics["fn_cost"],
                    "block_threshold": round(block_threshold, 4),
                    "review_threshold": round(review_threshold, 4),
                },
            }
        )

    # Stratified sample, not a plain top-N sort: block alone outnumbers the whole
    # sample, which would silently hide the review tier the 3-way routing exists to
    # produce. Take the highest-score rows *within each decision bucket* instead, so
    # the queue actually demonstrates all three routes an analyst will see.
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    by_decision = {"block": [], "review": [], "approve": []}
    for r in rows:
        by_decision[r["decision"]].append(r)
    bucket_caps = {"block": 90, "review": 130, "approve": 30}
    queue = [r for bucket in bucket_caps for r in by_decision[bucket][: bucket_caps[bucket]]]
    queue = queue[:QUEUE_SAMPLE_SIZE]

    (DASHBOARD_DATA_DIR / "queue.json").write_text(json.dumps(queue, indent=2))
    decision_counts = {d: len(by_decision[d]) for d in bucket_caps}
    print(f"Wrote queue.json ({len(queue)} sampled; full test-set decision split: {decision_counts})")

    # Ring graph: nodes/edges restricted to accounts in any flagged community, tagged
    # with both ground-truth ring id (if any) and detected ring id, so the dashboard can
    # render the "did detection match the injected ring" overlay honestly.
    flagged_accounts = set(ring_membership.keys())
    ground_truth_membership = {
        account: ring_id for ring_id, members in rings_ground_truth.items() for account in members
    }
    flagged_accounts |= set(ground_truth_membership.keys())

    graph_edges = edges[
        edges["entity_a"].isin(flagged_accounts) & edges["entity_b"].isin(flagged_accounts)
    ]
    nodes = [
        {
            "id": account,
            "detected_ring_id": ring_membership.get(account),
            "ground_truth_ring_id": ground_truth_membership.get(account),
        }
        for account in sorted(flagged_accounts)
    ]
    graph = {
        "nodes": nodes,
        # `weight` is the number of distinct shared entities (device/card/ip) linking
        # this pair — build_entity_edges aggregates edge_type into this count rather
        # than keeping one row per type, so weight=2 means e.g. "shares both a device
        # and a card," a stronger signal than weight=1.
        "edges": [
            {"source": r["entity_a"], "target": r["entity_b"], "weight": int(r["weight"])}
            for _, r in graph_edges.iterrows()
        ],
    }
    (DASHBOARD_DATA_DIR / "ring_graph.json").write_text(json.dumps(graph, indent=2))
    print(f"Wrote ring_graph.json ({len(nodes)} nodes, {len(graph['edges'])} edges)")

    system_health = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "point_risk_model": metrics,
        "ring_detection": ring_report,
        "decision_layer": decision_layer,
        "graph_cache_status": "fresh",  # placeholder for the Day 7 degraded-mode demo
    }
    (DASHBOARD_DATA_DIR / "system_health.json").write_text(json.dumps(system_health, indent=2))
    print("Wrote system_health.json")


if __name__ == "__main__":
    main()
