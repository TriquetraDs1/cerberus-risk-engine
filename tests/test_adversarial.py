"""Day 5-6 sanity tests. Deliberately independent of any trained model file — these
test the strategies and the search mechanism in isolation with a synthetic scoring
function, so they run fast and don't require the full pipeline to have been executed
first (consistent with this repo's "each stage independently testable" convention).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.adversarial.attacker import DetectionScore, adaptive_search
from cerberus.adversarial.strategies import (
    STRATEGY_APPLIERS,
    STRATEGY_PARAM_BOUNDS,
    apply_identity_rotation,
    apply_slow_ramp,
    apply_structuring,
    generate_baseline_ring,
)


def _rng():
    return np.random.default_rng(42)


def test_baseline_ring_has_expected_shape():
    ring = generate_baseline_ring(_rng(), ring_size=6)
    assert len(ring.member_account_ids) == 6
    assert set(ring.transactions["account_id"]) <= set(ring.member_account_ids)
    assert ring.transactions["label"].eq(1).all()
    # naive baseline: every transaction shares exactly one device
    assert ring.transactions["device_id"].nunique() == 1


def test_structuring_splits_amounts_and_multiplies_transaction_count():
    ring = generate_baseline_ring(_rng(), ring_size=4)
    original_total = ring.transactions["amount"].sum()
    original_count = len(ring.transactions)

    attacked = apply_structuring(ring, _rng(), n_splits=4, jitter=0.0)

    assert len(attacked.transactions) == original_count * 4
    # splitting preserves total spend (modulo the jitter, which is 0 here, and
    # per-split rounding to the nearest cent)
    assert attacked.transactions["amount"].sum() == pytest.approx(original_total, abs=0.5)


def test_identity_rotation_spreads_devices():
    ring = generate_baseline_ring(_rng(), ring_size=8)
    assert ring.transactions["device_id"].nunique() == 1

    attacked = apply_identity_rotation(ring, _rng(), n_devices=4)
    assert attacked.transactions["device_id"].nunique() == 4


def test_slow_ramp_stretches_time_window():
    ring = generate_baseline_ring(_rng(), ring_size=5)
    original_span = (ring.transactions["timestamp"].max() - ring.transactions["timestamp"].min()).total_seconds()

    attacked = apply_slow_ramp(ring, _rng(), ramp_multiplier=10.0)
    attacked_span = (attacked.transactions["timestamp"].max() - attacked.transactions["timestamp"].min()).total_seconds()

    # allow slack for the re-jitter step, but the span should be roughly 10x
    assert attacked_span > original_span * 5


def test_adaptive_search_moves_toward_lower_score():
    """A synthetic scorer where detection provably decreases as n_splits increases —
    confirms the search actually exploits a known-good direction, not just wandering.
    """

    def synthetic_scorer(ring) -> DetectionScore:
        n_splits = ring.params.get("n_splits", 1)
        score = max(0.0, 1.0 - 0.15 * (n_splits - 1))
        return DetectionScore(score, score, score)

    result = adaptive_search(
        strategy_name="structuring",
        applier=STRATEGY_APPLIERS["structuring"],
        param_bounds=STRATEGY_PARAM_BOUNDS["structuring"],
        make_base_ring=lambda: generate_baseline_ring(_rng(), ring_size=4),
        score_fn=synthetic_scorer,
        rng=_rng(),
        n_restarts=3,
        n_steps=10,
    )

    # the search should have found something meaningfully better than a random
    # single draw from the param space (baseline score, drawn once, is a fair
    # reference point for "no search at all")
    assert result.best_score.combined_score < result.baseline_score.combined_score
    assert result.best_params["n_splits"] > 1
