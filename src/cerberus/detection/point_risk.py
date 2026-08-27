"""Point-risk model: per-transaction fraud probability.

Day 1-2 baseline. Cost-sensitive threshold selection here is a preview of the Day 4
decision layer — real cost values belong there once the cost matrix work happens; the
placeholders below exist so this script prints something more honest than a bare
accuracy-maximizing threshold on day one.
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
from sklearn.model_selection import train_test_split

from cerberus.features.pipeline import FEATURE_COLUMNS

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - environment-dependent
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_LIGHTGBM = False


# Placeholder cost ratio: blocking a legitimate transaction (FP) vs. missing fraud (FN).
# Refine with real numbers in the Day 4 decision layer — this is intentionally a rough
# starting assumption (a typical review/chargeback cost asymmetry), not a tuned constant.
DEFAULT_FP_COST = 5.0
DEFAULT_FN_COST = 50.0


@dataclass
class TrainResult:
    model: object
    roc_auc: float
    pr_auc: float
    cost_optimal_threshold: float
    cost_at_optimal_threshold: float
    cost_at_default_threshold: float
    n_train: int
    n_test: int


def time_based_split(txns: pd.DataFrame, test_fraction: float = 0.2):
    """Split by time, not randomly — fraud rings cluster in short windows, so a random
    split would leak ring members between train and test and overstate performance.
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
            random_state=1337,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=1337
        )
    model.fit(X_train, y_train)
    return model


def cost_sensitive_threshold(
    y_true: np.ndarray, y_score: np.ndarray, fp_cost: float, fn_cost: float
) -> tuple[float, float]:
    """Sweep thresholds from the PR curve and return the one minimizing total expected
    cost = fp_cost * false_positives + fn_cost * false_negatives, plus that cost.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    best_threshold, best_cost = 0.5, float("inf")
    # precision_recall_curve returns thresholds of length len(precision)-1
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
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
    train_df, test_df = time_based_split(txns)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = _fit_classifier(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    best_threshold, best_cost = cost_sensitive_threshold(y_test.to_numpy(), scores, fp_cost, fn_cost)
    default_cost = cost_at_threshold(y_test.to_numpy(), scores, 0.5, fp_cost, fn_cost)

    return TrainResult(
        model=model,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        cost_optimal_threshold=best_threshold,
        cost_at_optimal_threshold=best_cost,
        cost_at_default_threshold=default_cost,
        n_train=len(train_df),
        n_test=len(test_df),
    )
