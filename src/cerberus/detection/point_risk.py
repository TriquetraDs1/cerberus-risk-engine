"""Point-risk model: per-transaction fraud probability.

Trains on a chronological split and reports a *calibrated* probability, not a bare
ranking score — see `cerberus.detection.calibration` for why that distinction matters.
The cost-sensitive threshold here is a preview of the Day 4 decision layer's global
threshold; per-segment thresholds live in `cerberus.decision.cost_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from cerberus.common.config import settings
from cerberus.detection.calibration import CalibrationReport, calibrate_and_report
from cerberus.features.pipeline import FEATURE_COLUMNS

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - environment-dependent
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_LIGHTGBM = False


# Placeholder global cost ratio: blocking a legitimate transaction (FP) vs. missing
# fraud (FN). Refined per-segment in cerberus.decision.cost_matrix (Day 4).
# Env-overridable via CERBERUS_FP_COST / CERBERUS_FN_COST.
DEFAULT_FP_COST = settings.fp_cost
DEFAULT_FN_COST = settings.fn_cost


@dataclass
class TrainResult:
    model: object
    calibration: CalibrationReport
    roc_auc: float
    pr_auc: float
    cost_optimal_threshold: float
    cost_at_optimal_threshold: float
    cost_at_default_threshold: float
    n_train: int
    n_calib: int
    n_test: int
    test_df: pd.DataFrame  # held-out rows, for downstream export/segmentation
    test_scores_calibrated: np.ndarray


def three_way_split(txns: pd.DataFrame, calib_fraction: float = 0.16, test_fraction: float = 0.2):
    """Chronological train / calibration / test split.

    Three-way, not two — the calibration split must be genuinely held out from
    training (so the isotonic map reflects real generalization error, not memorized
    training scores) AND disjoint from the test split (so "held-out" metrics aren't
    contaminated by data the calibrator already saw). Time-ordered, not random,
    for the same reason as Day 1-2's split: fraud rings cluster in short windows, and
    a random split would leak ring members across the boundary.
    """
    txns = txns.sort_values("timestamp")
    n = len(txns)
    train_end = int(n * (1 - calib_fraction - test_fraction))
    calib_end = int(n * (1 - test_fraction))
    return txns.iloc[:train_end], txns.iloc[train_end:calib_end], txns.iloc[calib_end:]


def time_based_split(txns: pd.DataFrame, test_fraction: float = 0.2):
    """Two-way chronological split (train / test only), used where a calibration
    split isn't needed — e.g. quick sanity checks outside the main training script.
    """
    txns = txns.sort_values("timestamp")
    cutoff = int(len(txns) * (1 - test_fraction))
    return txns.iloc[:cutoff], txns.iloc[cutoff:]


def _fit_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    if _HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=settings.random_seed,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=settings.random_seed
        )
    model.fit(X_train, y_train)
    return model


def cost_sensitive_threshold(
    y_true: np.ndarray, y_score: np.ndarray, fp_cost: float, fn_cost: float
) -> tuple[float, float]:
    """Sweep thresholds from the PR curve and return the one minimizing total expected
    cost = fp_cost * false_positives + fn_cost * false_negatives, plus that cost.
    """
    if len(y_true) == 0 or y_true.sum() == 0:
        # No positives in this slice (e.g. a small segment) — no threshold can trade
        # off recall against cost, so fall back to the neutral default rather than
        # returning a nonsensical "optimal" threshold from an empty sweep.
        return 0.5, cost_at_threshold(y_true, y_score, 0.5, fp_cost, fn_cost)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    best_threshold, best_cost = 0.5, float("inf")
    # precision_recall_curve returns thresholds of length len(precision)-1
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds, strict=True):
        tp = r * n_pos
        fn = n_pos - tp
        # precision = tp / (tp + fp)  =>  fp = tp * (1 - p) / p, guarding p == 0
        fp = tp * (1 - p) / p if p > 0 else n_neg
        cost = fp_cost * fp + fn_cost * fn
        if cost < best_cost:
            best_cost, best_threshold = cost, float(t)
    return best_threshold, best_cost


def cost_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float, fp_cost: float, fn_cost: float) -> float:
    preds = (y_score >= threshold).astype(int)
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return fp_cost * fp + fn_cost * fn


def train(
    txns: pd.DataFrame,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> TrainResult:
    train_df, calib_df, test_df = three_way_split(txns)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_calib, y_calib = calib_df[FEATURE_COLUMNS], calib_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = _fit_classifier(X_train, y_train)

    raw_calib_scores = model.predict_proba(X_calib)[:, 1]
    raw_test_scores = model.predict_proba(X_test)[:, 1]

    calibration = calibrate_and_report(
        raw_calib_scores, y_calib.to_numpy(), raw_test_scores, y_test.to_numpy()
    )
    calibrated_scores = calibration.calibrator.predict(raw_test_scores)

    # Isotonic calibration is monotonic, so ranking-based metrics are unchanged by
    # construction — computed on the calibrated scores anyway so every reported number
    # downstream traces back to the same score the model actually serves.
    roc_auc = roc_auc_score(y_test, calibrated_scores)
    pr_auc = average_precision_score(y_test, calibrated_scores)

    best_threshold, best_cost = cost_sensitive_threshold(
        y_test.to_numpy(), calibrated_scores, fp_cost, fn_cost
    )
    default_cost = cost_at_threshold(y_test.to_numpy(), calibrated_scores, 0.5, fp_cost, fn_cost)

    return TrainResult(
        model=model,
        calibration=calibration,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        cost_optimal_threshold=best_threshold,
        cost_at_optimal_threshold=best_cost,
        cost_at_default_threshold=default_cost,
        n_train=len(train_df),
        n_calib=len(calib_df),
        n_test=len(test_df),
        test_df=test_df,
        test_scores_calibrated=calibrated_scores,
    )
