#!/usr/bin/env python
"""Day 4: derive per-segment cost matrices and cost-optimal 3-way routing thresholds.

Usage:
    python scripts/generate_data.py
    python scripts/train_baseline.py
    python scripts/build_decision_layer.py
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cerberus.common.config import (
    REPORTS_DIR,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.decision.cost_matrix import build_segment_routing, derive_segment_cost_matrices
from cerberus.detection.point_risk import three_way_split, train
from cerberus.features.pipeline import build_features

DECISION_LAYER_JSON = REPORTS_DIR / "decision_layer.json"


def main() -> None:
    if not SYNTHETIC_TRANSACTIONS_CSV.exists():
        raise SystemExit("No data found — run `python scripts/generate_data.py` first.")

    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    features = build_features(txns, edges)

    print("Deriving per-segment cost matrices from the training split...")
    train_df, _, _ = three_way_split(features)
    cost_matrices = derive_segment_cost_matrices(train_df)
    for seg, cm in cost_matrices.items():
        print(f"  {seg:24s} mean_amount={cm.mean_amount:8.2f}  fp_cost={cm.fp_cost:7.2f}  fn_cost={cm.fn_cost:8.2f}")

    print("\nRetraining to get calibrated held-out scores (same seed, deterministic)...")
    result = train(features)

    print("Building per-segment 3-way routing...")
    routing = build_segment_routing(
        result.test_df, result.test_scores_calibrated, cost_matrices, result.cost_optimal_threshold
    )

    print("\n--- Per-segment routing report ---")
    total_savings_num, total_savings_den = 0.0, 0.0
    for seg, r in routing.items():
        print(
            f"  {seg:24s} block>={r.block_threshold:.3f}  review>={r.review_threshold:.3f}  "
            f"[{r.n_block} block / {r.n_review} review / {r.n_approve} approve]  "
            f"cost saved vs. global default: {r.cost_savings_pct:5.1f}%"
        )
        total_savings_num += r.cost_at_global_default_threshold - r.cost_at_optimal_threshold
        total_savings_den += r.cost_at_global_default_threshold
    overall_savings_pct = 100.0 * total_savings_num / total_savings_den if total_savings_den else 0.0
    print(f"\n=> Segmented routing saves {overall_savings_pct:.1f}% total cost vs. applying the single global threshold to every segment.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_default_threshold": result.cost_optimal_threshold,
        "overall_savings_pct_vs_global_threshold": overall_savings_pct,
        "segments": {
            seg: {
                "cost_matrix": asdict(r.cost_matrix),
                "block_threshold": r.block_threshold,
                "review_threshold": r.review_threshold,
                "n_transactions": r.n_transactions,
                "n_block": r.n_block,
                "n_review": r.n_review,
                "n_approve": r.n_approve,
                "cost_at_optimal_threshold": r.cost_at_optimal_threshold,
                "cost_at_global_default_threshold": r.cost_at_global_default_threshold,
                "cost_savings_pct": r.cost_savings_pct,
            }
            for seg, r in routing.items()
        },
        "limitations": [
            "Cost matrices are derived from mean transaction amount + a fixed retention-"
            "sensitivity multiplier per segment, not a calibrated business study.",
            "The review-band width (REVIEW_BAND_FACTOR) is a heuristic, not itself "
            "cost-optimized.",
        ],
    }
    DECISION_LAYER_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {DECISION_LAYER_JSON}")


if __name__ == "__main__":
    main()
