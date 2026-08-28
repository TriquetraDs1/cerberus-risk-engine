"""Probability calibration for the point-risk model.

A LightGBM binary-objective score is a good *ranking* out of the box (that's what
ROC-AUC/PR-AUC measure), but it is not automatically a trustworthy *probability* —
`class_weight="balanced"` in particular re-weights the training distribution, which
skews raw scores away from the true P(fraud). A fraud analyst reading "risk_score:
0.79" is entitled to assume that means something close to a 79% chance, not just "in
the upper decile of the ranking." Calibration is what closes that gap.

Implemented as isotonic regression fit on a held-out calibration split — not
`sklearn.calibration.CalibratedClassifierCV(cv="prefit")`, which was deprecated in
sklearn 1.6 in favor of a wrapper (`FrozenEstimator`) not worth pinning a hackathon
submission's dependency version to. A hand-rolled monotonic map is one line simpler
and exactly as correct: isotonic regression IS the calibration method
`CalibratedClassifierCV` uses internally for method="isotonic".

Isotonic regression is monotonic, so it never changes the model's ranking — ROC-AUC
and PR-AUC are unaffected. What it changes is whether 0.79 means 79%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationReport:
    calibrator: IsotonicRegression
    brier_before: float
    brier_after: float
    reliability_curve: list[dict]  # [{bin_center, predicted_mean, observed_rate, count}, ...]
    expected_calibration_error_before: float
    expected_calibration_error_after: float


def fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic map from raw model score -> calibrated P(fraud) on a held-out
    calibration split (never the test set — that would leak calibration information
    into the "held-out" metrics the rest of the pipeline reports).
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, y_true)
    return calibrator


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and observed frequency,
    weighted by bin population — the standard single-number calibration metric."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(observed - predicted)
    return float(ece)


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    curve = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        curve.append(
            {
                "bin_center": round(float((bin_edges[b] + bin_edges[b + 1]) / 2), 3),
                "predicted_mean": round(float(y_prob[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return curve


def calibrate_and_report(
    calib_raw_scores: np.ndarray,
    calib_labels: np.ndarray,
    test_raw_scores: np.ndarray,
    test_labels: np.ndarray,
) -> CalibrationReport:
    """Fit the calibrator on the calibration split, then report before/after
    calibration quality measured honestly on the held-out test split.
    """
    calibrator = fit_calibrator(calib_raw_scores, calib_labels)
    test_calibrated = calibrator.predict(test_raw_scores)

    brier_before = brier_score_loss(test_labels, test_raw_scores)
    brier_after = brier_score_loss(test_labels, test_calibrated)
    ece_before = _expected_calibration_error(test_labels, test_raw_scores)
    ece_after = _expected_calibration_error(test_labels, test_calibrated)
    curve = _reliability_curve(test_labels, test_calibrated)

    return CalibrationReport(
        calibrator=calibrator,
        brier_before=brier_before,
        brier_after=brier_after,
        reliability_curve=curve,
        expected_calibration_error_before=ece_before,
        expected_calibration_error_after=ece_after,
    )
