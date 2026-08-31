#!/usr/bin/env python
"""Roadmap B2: train the per-account sequence model and measure what it adds.

Reports three things, because only the third one settles whether this belongs in the
pipeline: the sequence model's own held-out metrics, the point-risk model's on the same
rows, and the metrics of an ensemble of the two. A second model that doesn't beat the
first one *in combination* is a second model you shouldn't ship.

Usage:
    python scripts/generate_data.py
    python scripts/detect_rings.py
    python scripts/train_baseline.py
    python scripts/train_sequence.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cerberus.common.config import (
    BASELINE_MODEL_PATH,
    CALIBRATOR_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
    settings,
)
from cerberus.detection.calibration import fit_calibrator
from cerberus.detection.point_risk import three_way_split
from cerberus.detection.sequence_risk import predict_scores, train_sequence_model
from cerberus.features.pipeline import FEATURE_COLUMNS, build_features
from cerberus.features.sequences import build_sequences

SEQUENCE_MODEL_PATH = MODELS_DIR / "sequence_risk.pt"
SEQUENCE_CALIBRATOR_PATH = MODELS_DIR / "sequence_risk_calibrator.joblib"
SEQUENCE_METRICS_JSON = REPORTS_DIR / "sequence_metrics.json"

# Weight on the sequence model in the ensemble. Low on purpose: the point-risk model
# stays primary because it is the explainable one (SHAP -> reason codes). This is a
# tie-breaking second opinion, not a co-equal vote.
ENSEMBLE_WEIGHT = 0.3


def main() -> None:
    for path in (SYNTHETIC_TRANSACTIONS_CSV, BASELINE_MODEL_PATH, CALIBRATOR_PATH):
        if not path.exists():
            raise SystemExit(f"Missing {path} — run the earlier pipeline scripts first.")

    print("Loading transactions and building sequences...")
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)

    X_seq, y_seq, ordered = build_sequences(txns)
    print(f"  {X_seq.shape[0]:,} windows x {X_seq.shape[1]} steps x {X_seq.shape[2]} features")

    # Same chronological boundaries as the point-risk model. `build_sequences` returns
    # rows in time order, so index positions map straight onto the split fractions —
    # and a fraud ring can't leak across the boundary any more than it can there.
    n = len(ordered)
    train_end = int(n * (1 - 0.16 - 0.2))
    calib_end = int(n * (1 - 0.2))

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_calib, y_calib = X_seq[train_end:calib_end], y_seq[train_end:calib_end]
    X_test, y_test = X_seq[calib_end:], y_seq[calib_end:]

    print(f"\nTraining GRU (train={len(y_train):,}, calib={len(y_calib):,}, test={len(y_test):,})...")
    result = train_sequence_model(
        X_train, y_train, X_test, y_test, random_seed=settings.random_seed
    )

    print("\n--- Sequence model, held out ---")
    print(f"ROC-AUC: {result.roc_auc:.4f}   PR-AUC: {result.pr_auc:.4f}")

    # Calibrate on the held-out calibration split, exactly as the point-risk model does.
    calibrator = fit_calibrator(predict_scores(result.model, X_calib), y_calib)
    seq_test_scores = calibrator.predict(predict_scores(result.model, X_test))

    # The point-risk model on the same rows, for a like-for-like comparison.
    features = build_features(txns, edges)
    _, _, point_test_df = three_way_split(features)
    booster = lgb.Booster(model_file=str(BASELINE_MODEL_PATH))
    point_calibrator = joblib.load(CALIBRATOR_PATH)
    point_scores = point_calibrator.predict(booster.predict(point_test_df[FEATURE_COLUMNS]))
    point_y = point_test_df["label"].to_numpy()

    print("\n--- Point-risk model, same held-out window ---")
    print(f"ROC-AUC: {roc_auc_score(point_y, point_scores):.4f}   PR-AUC: {average_precision_score(point_y, point_scores):.4f}")

    # Both splits are the last 20% of the same time-ordered table, so they cover the same
    # rows; guard the length anyway rather than silently comparing misaligned arrays.
    ensemble_auc = ensemble_pr = None
    if len(point_scores) == len(seq_test_scores):
        ensemble = (1 - ENSEMBLE_WEIGHT) * point_scores + ENSEMBLE_WEIGHT * seq_test_scores
        ensemble_auc = float(roc_auc_score(point_y, ensemble))
        ensemble_pr = float(average_precision_score(point_y, ensemble))
        print(f"\n--- Ensemble ({1 - ENSEMBLE_WEIGHT:.0%} point-risk / {ENSEMBLE_WEIGHT:.0%} sequence) ---")
        print(f"ROC-AUC: {ensemble_auc:.4f}   PR-AUC: {ensemble_pr:.4f}")

        point_pr = average_precision_score(point_y, point_scores)
        if ensemble_pr > point_pr:
            print(f"\n=> The ensemble beats point-risk alone on PR-AUC ({ensemble_pr:.4f} > {point_pr:.4f}).")
        else:
            print(
                f"\n=> The ensemble does NOT beat point-risk alone on PR-AUC "
                f"({ensemble_pr:.4f} <= {point_pr:.4f}). Report that, and do not wire the "
                "sequence model into the decision layer on the strength of its solo score."
            )
    else:
        print(
            f"\n! Split sizes differ ({len(point_scores)} vs {len(seq_test_scores)}) — "
            "skipping the ensemble comparison rather than aligning them by assumption."
        )

    import torch

    torch.save(result.model.state_dict(), SEQUENCE_MODEL_PATH)
    joblib.dump(calibrator, SEQUENCE_CALIBRATOR_PATH)
    SEQUENCE_METRICS_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sequence_length": int(X_seq.shape[1]),
                "n_features_per_step": int(X_seq.shape[2]),
                "n_train": result.n_train,
                "n_test": result.n_test,
                "epochs": result.epochs_run,
                "final_train_loss": result.final_train_loss,
                "sequence_roc_auc": result.roc_auc,
                "sequence_pr_auc": result.pr_auc,
                "point_risk_roc_auc": float(roc_auc_score(point_y, point_scores)),
                "point_risk_pr_auc": float(average_precision_score(point_y, point_scores)),
                "ensemble_weight_sequence": ENSEMBLE_WEIGHT,
                "ensemble_roc_auc": ensemble_auc,
                "ensemble_pr_auc": ensemble_pr,
                "limitations": [
                    "The sequence model has no SHAP-equivalent explanation, so it cannot "
                    "produce reason codes. It is a second opinion, not a decision-maker — "
                    "docs/ARCHITECTURE.md requires a reason code on every block.",
                    "The ensemble weight is a chosen constant, not a fitted one. Fitting "
                    "it would need a fourth split to avoid selecting on the test set.",
                    "Windows are fixed-length and left-padded; an account's first "
                    "transactions are scored on mostly padding.",
                ],
            },
            indent=2,
        )
    )
    print(f"\nSaved model to {SEQUENCE_MODEL_PATH}")
    print(f"Saved calibrator to {SEQUENCE_CALIBRATOR_PATH}")
    print(f"Wrote {SEQUENCE_METRICS_JSON}")


if __name__ == "__main__":
    main()
