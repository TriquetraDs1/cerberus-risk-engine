#!/usr/bin/env python
"""Day 3: run Louvain ring detection and validate it against ground truth.

Usage:
    python scripts/generate_data.py   # first, if you haven't
    python scripts/detect_rings.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cerberus.common.config import (
    DETECTED_RINGS_JSON,
    RING_DETECTION_REPORT_JSON,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_HOUSEHOLD_PAIRS_JSON,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.detection.ring_detector import (
    COORDINATION_THRESHOLD,
    build_graph,
    detect_communities,
    evaluate_against_ground_truth,
)


def main() -> None:
    if not SYNTHETIC_ENTITY_EDGES_CSV.exists():
        raise SystemExit("No data found — run `python scripts/generate_data.py` first.")

    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    rings_ground_truth = json.loads(SYNTHETIC_RINGS_JSON.read_text())
    household_pairs = [tuple(p) for p in json.loads(SYNTHETIC_HOUSEHOLD_PAIRS_JSON.read_text())]

    print(f"Building entity-link graph from {len(edges):,} edges...")
    graph = build_graph(edges)
    print(f"  {graph.number_of_nodes():,} accounts, {graph.number_of_edges():,} unique links")

    print("Running Louvain community detection...")
    result = detect_communities(graph, transactions=txns)
    n_candidates = sum(1 for m in result.communities.values() if len(m) >= 3)
    print(f"  {len(result.communities):,} communities found")
    print(f"  {n_candidates} of size >= 3 (structural candidates)")
    print(f"  {len(result.flagged_ring_ids)} of those also look behaviourally coordinated -> flagged as rings")
    if result.coordination:
        scores = sorted(result.coordination.values(), reverse=True)
        print(
            f"  coordination score across candidates: max {scores[0]:.2f}, "
            f"median {scores[len(scores) // 2]:.2f}, min {scores[-1]:.2f}"
        )

    print("\nValidating against synthetic ground truth...")
    report = evaluate_against_ground_truth(result, rings_ground_truth, household_pairs)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    # The coordination threshold is a choice, so ship the curve it was chosen from. A
    # single operating point with no alternatives shown is an assertion; the curve lets a
    # reader see what was traded away and pick differently.
    if result.coordination:
        report["coordination_threshold"] = COORDINATION_THRESHOLD
        report["coordination_scores"] = {str(k): round(v, 4) for k, v in result.coordination.items()}
        ring_accounts = {a for members in rings_ground_truth.values() for a in members}
        curve = []
        for t in (0.24, 0.26, 0.28, 0.30, 0.32):
            caught = innocent = 0
            for cid, score in result.coordination.items():
                members = result.communities[cid]
                is_ring = sum(m in ring_accounts for m in members) / len(members) > 0.5
                if score >= t:
                    caught += is_ring
                    innocent += not is_ring
            curve.append(
                {"threshold": t, "ring_communities_flagged": caught, "innocent_communities_flagged": innocent}
            )
        report["coordination_threshold_curve"] = curve

    print("\n--- Honest ring-detection report ---")
    print(f"Injected rings:                {report['n_rings']}")
    print(f"Perfectly recovered (100%):    {report['n_perfectly_recovered']} / {report['n_rings']}")
    print(f"Mean ring recovery:            {report['mean_ring_recovery']:.1%}")
    print(
        f"\nInnocent household-sharing pairs: {report['n_household_pairs']}"
    )
    print(
        f"  wrongly co-flagged as a ring:   {report['n_household_false_positives']} "
        f"({report['household_false_positive_rate']:.1%} false-positive rate)"
    )
    print(
        f"\n=> This is the honest FP-cost story for the graph layer (see "
        f"docs/ARCHITECTURE.md, panel pushback #2): {report['household_false_positive_rate']:.1%} "
        f"of legitimate device-sharing would be wrongly escalated at min_ring_size=3."
    )

    # Detected rings, keyed like the ground truth file so a reviewer can diff them by eye.
    detected_rings = {
        f"detected_{cid}": members
        for cid, members in result.communities.items()
        if cid in result.flagged_ring_ids
    }
    DETECTED_RINGS_JSON.write_text(json.dumps(detected_rings, indent=2))
    RING_DETECTION_REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {DETECTED_RINGS_JSON}")
    print(f"Wrote {RING_DETECTION_REPORT_JSON}")


if __name__ == "__main__":
    main()
