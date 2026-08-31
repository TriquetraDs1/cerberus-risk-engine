#!/usr/bin/env python
"""Roadmap B1: train the GraphSAGE ring detector and compare it to Louvain head-to-head.

The comparison is the point of the script. Louvain's numbers (25/25 recovered, 9.3%
household false-positive rate) are already in reports/ring_detection_report.json; this
measures the learned detector on the *same* graph and the *same* household FP set, so
"would a GNN do better" gets an answer instead of an opinion.

Usage:
    python scripts/generate_data.py
    python scripts/detect_rings.py
    python scripts/train_gnn_rings.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cerberus.common.config import (
    MODELS_DIR,
    REPORTS_DIR,
    RING_DETECTION_REPORT_JSON,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_HOUSEHOLD_PAIRS_JSON,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
    settings,
)
from cerberus.detection.gnn_ring import (
    NODE_FEATURE_NAMES,
    build_edge_index,
    build_labels,
    build_node_features,
    predict_ring_scores,
    temporal_node_split,
    train_gnn,
)
from cerberus.detection.ring_detector import build_graph

GNN_MODEL_PATH = MODELS_DIR / "gnn_ring.pt"
GNN_METRICS_JSON = REPORTS_DIR / "gnn_ring_metrics.json"

# Score above which the GNN calls an account a ring member. 0.5 is the neutral choice for
# a class-weighted sigmoid; it is not cost-optimised, and the report says so.
FLAG_THRESHOLD = 0.5


def main() -> None:
    for path in (SYNTHETIC_ENTITY_EDGES_CSV, SYNTHETIC_RINGS_JSON, SYNTHETIC_TRANSACTIONS_CSV):
        if not path.exists():
            raise SystemExit(f"Missing {path} — run generate_data.py and detect_rings.py first.")

    print("Building the entity graph...")
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    rings = json.loads(SYNTHETIC_RINGS_JSON.read_text())

    graph = build_graph(edges)
    features, node_index = build_node_features(graph)
    edge_index = build_edge_index(graph, node_index)
    labels = build_labels(node_index, rings)
    print(
        f"  {len(node_index):,} nodes, {edge_index.shape[1] // 2:,} edges, "
        f"{int(labels.sum())} ring-member nodes, {len(NODE_FEATURE_NAMES)} features/node"
    )

    train_mask, test_mask = temporal_node_split(node_index, txns)
    print(
        f"  temporal split: {int(train_mask.sum()):,} train / {int(test_mask.sum()):,} test nodes "
        f"({int(labels[test_mask].sum())} ring members held out)"
    )

    print("\nTraining GraphSAGE...")
    result = train_gnn(
        features, edge_index, labels, train_mask, test_mask, random_seed=settings.random_seed
    )

    print("\n--- GNN, held-out nodes ---")
    print(f"ROC-AUC: {result.roc_auc:.4f}   PR-AUC: {result.pr_auc:.4f}")

    scores = predict_ring_scores(result.model, features, edge_index)
    flagged = {account for account, i in node_index.items() if scores[i] >= FLAG_THRESHOLD}

    # Same ground truth Louvain is scored against: per-ring recovery, and the honest
    # false-positive rate on innocent households that legitimately share a device.
    per_ring_recovery = {
        ring_id: (sum(m in flagged for m in members) / len(members) if members else 0.0)
        for ring_id, members in rings.items()
    }
    mean_recovery = float(np.mean(list(per_ring_recovery.values()))) if per_ring_recovery else 0.0
    n_perfect = sum(1 for v in per_ring_recovery.values() if v == 1.0)

    household_fp_rate = None
    n_household_fp = None
    if SYNTHETIC_HOUSEHOLD_PAIRS_JSON.exists():
        pairs = json.loads(SYNTHETIC_HOUSEHOLD_PAIRS_JSON.read_text())
        n_household_fp = sum(1 for a, b in pairs if a in flagged and b in flagged)
        household_fp_rate = n_household_fp / len(pairs) if pairs else 0.0

    print(f"\n--- GNN vs. ground truth (threshold {FLAG_THRESHOLD}) ---")
    print(f"Perfectly recovered rings: {n_perfect} / {len(rings)}")
    print(f"Mean ring recovery:        {mean_recovery:.1%}")
    if household_fp_rate is not None:
        print(f"Household FP rate:         {n_household_fp}/{len(pairs)} ({household_fp_rate:.1%})")

    # A perfect held-out score is a reason for suspicion, not celebration. Injected rings
    # are dense cliques (degree ~5-11) while innocent household pairs are single links
    # (degree 1), so the task may be separable by one feature — in which case the GNN's
    # message passing contributed nothing and calling this a "GNN win" would be false.
    # This is the check that distinguishes the two, and it runs every time.
    degrees = features[:, NODE_FEATURE_NAMES.index("degree")]
    best_degree_recovery, best_degree_fp, best_degree_cut, best_separation = 0.0, None, None, -1.0
    for cut in range(1, int(degrees.max()) + 1):
        degree_flagged = {a for a, i in node_index.items() if degrees[i] >= cut}
        recovery = float(
            np.mean(
                [
                    sum(m in degree_flagged for m in members) / len(members)
                    for members in rings.values()
                    if members
                ]
            )
        )
        fp = (
            sum(1 for a, b in pairs if a in degree_flagged and b in degree_flagged) / len(pairs)
            if SYNTHETIC_HOUSEHOLD_PAIRS_JSON.exists() and pairs
            else 0.0
        )
        # Score the threshold on separation, not recall: degree >= 1 flags everything,
        # scoring 100% recovery and 100% false positives, and would "win" a
        # recovery-only search while detecting nothing.
        separation = recovery - fp
        if separation > best_separation:
            best_separation, best_degree_recovery, best_degree_fp, best_degree_cut = (
                separation,
                recovery,
                fp,
                cut,
            )

    print(f"\n--- Trivial baseline: flag every node with degree >= {best_degree_cut} ---")
    print(f"Mean ring recovery:        {best_degree_recovery:.1%}")
    if best_degree_fp is not None:
        print(f"Household FP rate:         {best_degree_fp:.1%}")
    if best_degree_recovery >= mean_recovery and (best_degree_fp or 0) <= (household_fp_rate or 0):
        print(
            "\n=> A one-line degree threshold matches the GNN here. The graph structure "
            "in this synthetic data is separable without message passing, so these GNN "
            "numbers measure the dataset's easiness, not the model's power. Say that "
            "before anyone asks."
        )

    louvain = json.loads(RING_DETECTION_REPORT_JSON.read_text()) if RING_DETECTION_REPORT_JSON.exists() else None
    if louvain:
        print("\n--- Head to head ---")
        print(f"{'':<22}{'Louvain':>10}{'GNN':>10}")
        print(f"{'perfectly recovered':<22}{louvain['n_perfectly_recovered']:>10}{n_perfect:>10}")
        print(f"{'mean recovery':<22}{louvain['mean_ring_recovery']:>10.1%}{mean_recovery:>10.1%}")
        if household_fp_rate is not None:
            print(
                f"{'household FP rate':<22}{louvain['household_false_positive_rate']:>10.1%}"
                f"{household_fp_rate:>10.1%}"
            )
        if mean_recovery < louvain["mean_ring_recovery"]:
            print(
                "\n=> Louvain still wins on recovery. Report it that way: the learned "
                "detector did not beat the unsupervised one on this dataset, and the "
                "explainable method remains the default on merit, not on preference."
            )

    import torch

    torch.save(result.model.state_dict(), GNN_MODEL_PATH)
    GNN_METRICS_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_nodes": len(node_index),
                "n_edges": int(edge_index.shape[1] // 2),
                "node_features": NODE_FEATURE_NAMES,
                "n_train_nodes": result.n_train_nodes,
                "n_test_nodes": result.n_test_nodes,
                "epochs": result.epochs_run,
                "final_loss": result.final_loss,
                "flag_threshold": FLAG_THRESHOLD,
                "gnn_roc_auc": result.roc_auc,
                "gnn_pr_auc": result.pr_auc,
                "gnn_n_perfectly_recovered": n_perfect,
                "gnn_mean_ring_recovery": mean_recovery,
                "gnn_household_false_positive_rate": household_fp_rate,
                "degree_baseline": {
                    "threshold": best_degree_cut,
                    "mean_ring_recovery": best_degree_recovery,
                    "household_false_positive_rate": best_degree_fp,
                },
                "louvain_comparison": louvain,
                "limitations": [
                    "Transductive: message passing sees the whole graph at inference. A "
                    "production detector would need the inductive setting, where a new "
                    "account arrives without its neighbourhood already in the graph.",
                    "The node split is temporal by first-transaction time, which holds "
                    "out late-forming rings, but a held-out node's neighbours may still "
                    "be training nodes — unavoidable in one connected graph, and it "
                    "makes these numbers optimistic relative to a true cold start.",
                    "The flag threshold is 0.5, not cost-optimised the way the "
                    "point-risk decision layer's per-segment thresholds are.",
                    "Ring labels come from the synthetic generator; this measures "
                    "learnability of injected structure, not real fraud rings.",
                    "Compare gnn_* against degree_baseline before quoting any GNN result: "
                    "if a single degree threshold matches it, these numbers describe how "
                    "separable the synthetic graph is, not what the GNN learned.",
                ],
            },
            indent=2,
        )
    )
    print(f"\nSaved model to {GNN_MODEL_PATH}")
    print(f"Wrote {GNN_METRICS_JSON}")


if __name__ == "__main__":
    main()
