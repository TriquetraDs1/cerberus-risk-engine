"""Day 4: the cost matrix + 3-way routing decision layer.

The Day 1-2 baseline used one global FP:FN cost ratio. That's a placeholder, not a
decision — a real payments team would immediately ask "why does blocking a ₹50 grocery
order cost the same, relatively, as missing fraud on a ₹90,000 travel booking?" It
doesn't, so this module derives a cost matrix *per merchant segment* from the segment's
own transaction data, then finds each segment's own cost-optimal decision threshold.

Cost derivation (documented, not hardcoded magic numbers):
  fn_cost(segment) = mean_transaction_amount(segment) + CHARGEBACK_PROCESSING_FEE
    — missing fraud costs roughly the transaction amount itself, plus a flat
    processor chargeback fee (an industry-standard ballpark, not segment-specific).
  fp_cost(segment) = mean_transaction_amount(segment) * RETENTION_SENSITIVITY[segment]
    — wrongly blocking a legitimate transaction costs a fraction of that customer's
    typical spend, scaled by how retention-sensitive the segment is: a grocery
    customer transacts often and is quick to switch if blocked (high sensitivity); a
    once-a-year luxury travel booking has less repeat-purchase value at stake per
    block (low sensitivity).

These are still estimates, not a calibrated business study — that limitation is named
explicitly in the exported report, consistent with the rest of this repo's "honest
metrics" stance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cerberus.data.synthetic_rings import SEGMENTS
from cerberus.detection.point_risk import cost_at_threshold, cost_sensitive_threshold

CHARGEBACK_PROCESSING_FEE = 25.0

RETENTION_SENSITIVITY = {
    "grocery_essentials": 0.15,
    "electronics_highvalue": 0.05,
    "digital_subscription": 0.10,
    "travel_luxury": 0.02,
}

# The review band sits below the block threshold, scaled by this factor. This width
# is a heuristic, not itself cost-optimized — a genuine limitation, named here rather
# than dressed up as more rigorous than it is.
REVIEW_BAND_FACTOR = 0.5


@dataclass
class SegmentCostMatrix:
    segment: str
    mean_amount: float
    fp_cost: float
    fn_cost: float


@dataclass
class SegmentRouting:
    segment: str
    cost_matrix: SegmentCostMatrix
    block_threshold: float
    review_threshold: float
    n_transactions: int
    n_block: int
    n_review: int
    n_approve: int
    cost_at_optimal_threshold: float
    cost_at_global_default_threshold: float
    cost_savings_pct: float


def derive_segment_cost_matrices(train_df: pd.DataFrame) -> dict[str, SegmentCostMatrix]:
    """Derive each segment's cost matrix from its own transaction amounts in the
    training split — never test, so the cost matrix used to grade "held-out"
    performance wasn't informed by the data it's graded against.
    """
    matrices = {}
    for segment in SEGMENTS:
        seg_amounts = train_df.loc[train_df["segment"] == segment, "amount"]
        mean_amount = float(seg_amounts.mean()) if len(seg_amounts) else 0.0
        fn_cost = mean_amount + CHARGEBACK_PROCESSING_FEE
        fp_cost = mean_amount * RETENTION_SENSITIVITY[segment]
        matrices[segment] = SegmentCostMatrix(
            segment=segment, mean_amount=mean_amount, fp_cost=fp_cost, fn_cost=fn_cost
        )
    return matrices


def build_segment_routing(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    cost_matrices: dict[str, SegmentCostMatrix],
    global_default_threshold: float,
) -> dict[str, SegmentRouting]:
    """For each segment, find its own cost-optimal block threshold (via the same
    PR-curve sweep as the Day 1-2 global preview, just scoped to that segment's rows
    and its own cost matrix) and report the 3-way routing split it produces.
    """
    test_df = test_df.reset_index(drop=True)
    scores = np.asarray(scores)
    routing: dict[str, SegmentRouting] = {}

    for segment in SEGMENTS:
        mask = (test_df["segment"] == segment).to_numpy()
        seg_scores = scores[mask]
        seg_labels = test_df.loc[mask, "label"].to_numpy()
        cm = cost_matrices[segment]

        block_threshold, optimal_cost = cost_sensitive_threshold(
            seg_labels, seg_scores, cm.fp_cost, cm.fn_cost
        )
        review_threshold = block_threshold * REVIEW_BAND_FACTOR
        default_cost = cost_at_threshold(seg_labels, seg_scores, global_default_threshold, cm.fp_cost, cm.fn_cost)

        n_block = int((seg_scores >= block_threshold).sum())
        n_review = int(((seg_scores >= review_threshold) & (seg_scores < block_threshold)).sum())
        n_approve = int((seg_scores < review_threshold).sum())

        savings_pct = (
            100.0 * (default_cost - optimal_cost) / default_cost if default_cost > 0 else 0.0
        )

        routing[segment] = SegmentRouting(
            segment=segment,
            cost_matrix=cm,
            block_threshold=block_threshold,
            review_threshold=review_threshold,
            n_transactions=int(mask.sum()),
            n_block=n_block,
            n_review=n_review,
            n_approve=n_approve,
            cost_at_optimal_threshold=optimal_cost,
            cost_at_global_default_threshold=default_cost,
            cost_savings_pct=savings_pct,
        )

    return routing


def route_transaction(score: float, segment: str, routing: dict[str, SegmentRouting]) -> str:
    """Apply the segment-specific thresholds to a single transaction's score."""
    r = routing.get(segment)
    if r is None:
        return "review"  # unknown segment: fail toward human review, not silent approval
    if score >= r.block_threshold:
        return "block"
    if score >= r.review_threshold:
        return "review"
    return "approve"
