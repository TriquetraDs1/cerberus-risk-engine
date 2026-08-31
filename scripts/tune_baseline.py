#!/usr/bin/env python
"""Hyperparameter search for the point-risk booster — optimising COST, not AUC.

The objective is total expected cost at the cost-optimal threshold, evaluated on the
held-out *calibration* split. Two deliberate choices there:

  * **Cost, not ROC-AUC.** This project's whole argument is that a fraud model should be
    judged by what its mistakes cost, not by a ranking metric (docs/ARCHITECTURE.md §1).
    Tuning for AUC and then reporting cost would be exactly the incoherence it criticises.
  * **The calibration split, not the test split.** The test split has to stay untouched by
    every fitting decision, and hyperparameter selection is a fitting decision. Tuning on
    test would leak, and the reported held-out numbers would be optimistic.

Scores are calibrated before the cost sweep, so the objective is measured on the same
kind of number the product actually serves.

Usage:
    python scripts/tune_baseline.py                  # 40 trials (default)
    python scripts/tune_baseline.py --n-trials 100
    python scripts/tune_baseline.py --timeout 900    # stop after 15 minutes

Writes reports/tuning_study.json. Copy the printed `best_params` into
`LGBM_PARAMS` in src/cerberus/detection/point_risk.py, then re-run the pipeline.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna
import pandas as pd

from cerberus.common.config import (
    REPORTS_DIR,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
    settings,
)
from cerberus.detection.calibration import fit_calibrator
from cerberus.detection.point_risk import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    LGBM_PARAMS,
    cost_sensitive_threshold,
    fit_classifier,
    three_way_split,
)
from cerberus.features.pipeline import FEATURE_COLUMNS, build_features

TUNING_REPORT_JSON = REPORTS_DIR / "tuning_study.json"


def _objective_factory(X_train, y_train, X_calib, y_calib):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = fit_classifier(X_train, y_train, params)
        raw = model.predict_proba(X_calib)[:, 1]
        # Calibrate before costing: an uncalibrated score would make the threshold
        # sweep compare probabilities that don't mean the same thing across trials.
        calibrator = fit_calibrator(raw, y_calib.to_numpy())
        scores = calibrator.predict(raw)
        _, cost = cost_sensitive_threshold(
            y_calib.to_numpy(), scores, DEFAULT_FP_COST, DEFAULT_FN_COST
        )
        return cost

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None, help="seconds; stops early if set")
    args = parser.parse_args()

    if not SYNTHETIC_TRANSACTIONS_CSV.exists():
        raise SystemExit(f"Missing {SYNTHETIC_TRANSACTIONS_CSV} — run scripts/generate_data.py first.")

    print("Loading transactions and building features...")
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    features = build_features(txns, edges)
    train_df, calib_df, _ = three_way_split(features)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_calib, y_calib = calib_df[FEATURE_COLUMNS], calib_df["label"]

    # Baseline: what the current hardcoded params cost, so the study's improvement is
    # reported against a real number rather than just "the best trial we happened to see".
    baseline_model = fit_classifier(X_train, y_train, LGBM_PARAMS)
    baseline_raw = baseline_model.predict_proba(X_calib)[:, 1]
    baseline_calibrator = fit_calibrator(baseline_raw, y_calib.to_numpy())
    _, baseline_cost = cost_sensitive_threshold(
        y_calib.to_numpy(),
        baseline_calibrator.predict(baseline_raw),
        DEFAULT_FP_COST,
        DEFAULT_FN_COST,
    )
    print(f"Current LGBM_PARAMS cost on the calibration split: {baseline_cost:,.0f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
        study_name="cerberus-point-risk-cost",
    )
    print(f"Running {args.n_trials} trials (objective: cost at the cost-optimal threshold)...")
    study.optimize(
        _objective_factory(X_train, y_train, X_calib, y_calib),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    improvement_pct = (
        100.0 * (baseline_cost - study.best_value) / baseline_cost if baseline_cost else 0.0
    )

    print("\n--- Tuning result ---")
    print(f"Baseline cost: {baseline_cost:,.0f}")
    print(f"Best cost:     {study.best_value:,.0f}   ({improvement_pct:+.1f}% vs. baseline)")
    print("\nBest params (copy into LGBM_PARAMS in src/cerberus/detection/point_risk.py):")
    print(json.dumps(study.best_params, indent=2))

    if improvement_pct <= 0:
        print(
            "\n=> The search did not beat the current params. Keep LGBM_PARAMS as they are "
            "and report that honestly — a null result is a result."
        )

    TUNING_REPORT_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "objective": "cost at the cost-optimal threshold, calibration split",
                "fp_cost": DEFAULT_FP_COST,
                "fn_cost": DEFAULT_FN_COST,
                "n_trials": len(study.trials),
                "baseline_params": LGBM_PARAMS,
                "baseline_cost": baseline_cost,
                "best_params": study.best_params,
                "best_cost": study.best_value,
                "improvement_pct_vs_baseline": improvement_pct,
                "limitations": [
                    "Tuned on the calibration split, which is also what the isotonic map is "
                    "fit on — the split does double duty here. A fourth split would be "
                    "cleaner; at this dataset size it would cost more in variance than it "
                    "buys in independence.",
                    "The objective uses the global FP/FN cost preview, not the per-segment "
                    "cost matrices of the Day 4 decision layer.",
                ],
            },
            indent=2,
        )
    )
    print(f"\nWrote {TUNING_REPORT_JSON}")


if __name__ == "__main__":
    main()
