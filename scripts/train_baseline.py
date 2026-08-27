#!/usr/bin/env python
"""Day 1-2: train the baseline point-risk model and print an honest metrics report.

Usage:
    python scripts/generate_data.py   # first, if you haven't
    python scripts/train_baseline.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cerberus.common.config import (
    BASELINE_METRICS_JSON,
    BASELINE_MODEL_PATH,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.detection.point_risk import DEFAULT_FN_COST, DEFAULT_FP_COST, train
from cerberus.features.pipeline import build_features


def main() -> None:
    if not SYNTHETIC_TRANSACTIONS_CSV.exists():
        raise SystemExit("No data found — run `python scripts/generate_data.py` first.")

    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)

    print(f"Loaded {len(txns):,} transactions. Building features...")
    features = build_features(txns, edges)

    print("Training point-risk model (time-based train/test split)...")
    result = train(features, fp_cost=DEFAULT_FP_COST, fn_cost=DEFAULT_FN_COST)

    print("\n--- Honest metrics report ---")
    print(f"train / test size:        {result.n_train:,} / {result.n_test:,}")
    print(f"ROC-AUC:                  {result.roc_auc:.4f}")
    print(f"PR-AUC (avg precision):   {result.pr_auc:.4f}")
    print(
        f"\nCost model (placeholder — refine in Day 4 decision layer): "
        f"FP cost={DEFAULT_FP_COST}, FN cost={DEFAULT_FN_COST}"
    )
    print(f"  cost at default 0.5 threshold:      {result.cost_at_default_threshold:,.0f}")
    print(
        f"  cost at cost-optimal threshold "
        f"({result.cost_optimal_threshold:.3f}): {result.cost_at_optimal_threshold:,.0f}"
    )
    savings = result.cost_at_default_threshold - result.cost_at_optimal_threshold
    print(f"  => cost-aware thresholding saves {savings:,.0f} cost units over the naive 0.5 cutoff")

    if hasattr(result.model, "booster_"):
        result.model.booster_.save_model(str(BASELINE_MODEL_PATH))
        print(f"\nSaved model to {BASELINE_MODEL_PATH}")
    else:
        import joblib

        joblib.dump(result.model, str(BASELINE_MODEL_PATH.with_suffix(".joblib")))
        print(f"\nSaved model to {BASELINE_MODEL_PATH.with_suffix('.joblib')}")

    # Real numbers for the dashboard to render — no hand-typed mock metrics.
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "fp_cost": DEFAULT_FP_COST,
        "fn_cost": DEFAULT_FN_COST,
        "cost_optimal_threshold": result.cost_optimal_threshold,
        "cost_at_optimal_threshold": result.cost_at_optimal_threshold,
        "cost_at_default_threshold": result.cost_at_default_threshold,
    }
    BASELINE_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {BASELINE_METRICS_JSON}")


if __name__ == "__main__":
    main()
