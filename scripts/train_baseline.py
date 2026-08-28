#!/usr/bin/env python
"""Day 1-2 baseline + calibration: train the point-risk model and print an honest,
calibrated metrics report.

Usage:
    python scripts/generate_data.py   # first, if you haven't
    python scripts/train_baseline.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from cerberus.common.config import (
    BASELINE_METRICS_JSON,
    BASELINE_MODEL_PATH,
    CALIBRATOR_PATH,
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

    print("Training point-risk model (train / calibration / test chronological split)...")
    result = train(features, fp_cost=DEFAULT_FP_COST, fn_cost=DEFAULT_FN_COST)
    cal = result.calibration

    print("\n--- Honest metrics report ---")
    print(f"train / calib / test size: {result.n_train:,} / {result.n_calib:,} / {result.n_test:,}")
    print(f"ROC-AUC:                  {result.roc_auc:.4f}")
    print(f"PR-AUC (avg precision):   {result.pr_auc:.4f}")

    print("\n--- Calibration ---")
    print(f"Brier score  (lower is better):  raw={cal.brier_before:.4f}  ->  calibrated={cal.brier_after:.4f}")
    print(f"Expected calibration error:      raw={cal.expected_calibration_error_before:.4f}  ->  calibrated={cal.expected_calibration_error_after:.4f}")
    if cal.brier_after > cal.brier_before:
        print(
            "  NOTE: calibration did not improve Brier score on this run — reporting it "
            "anyway rather than hiding it. See docs/ARCHITECTURE.md limitations."
        )

    print(
        f"\nCost model (global preview — refined per-segment in Day 4 decision layer): "
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
        joblib.dump(result.model, str(BASELINE_MODEL_PATH.with_suffix(".joblib")))
        print(f"\nSaved model to {BASELINE_MODEL_PATH.with_suffix('.joblib')}")

    joblib.dump(cal.calibrator, str(CALIBRATOR_PATH))
    print(f"Saved calibrator to {CALIBRATOR_PATH}")

    # Real numbers for the dashboard to render — no hand-typed mock metrics.
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_train": result.n_train,
        "n_calib": result.n_calib,
        "n_test": result.n_test,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "fp_cost": DEFAULT_FP_COST,
        "fn_cost": DEFAULT_FN_COST,
        "cost_optimal_threshold": result.cost_optimal_threshold,
        "cost_at_optimal_threshold": result.cost_at_optimal_threshold,
        "cost_at_default_threshold": result.cost_at_default_threshold,
        "calibration": {
            "brier_before": cal.brier_before,
            "brier_after": cal.brier_after,
            "expected_calibration_error_before": cal.expected_calibration_error_before,
            "expected_calibration_error_after": cal.expected_calibration_error_after,
            "reliability_curve": cal.reliability_curve,
        },
    }
    BASELINE_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {BASELINE_METRICS_JSON}")


if __name__ == "__main__":
    main()
