"""Sanity tests for the synthetic generator — mainly: does the ground truth we claim to
inject actually show up in the data, so Day 3's Louvain layer has something real to find.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.data.synthetic_rings import GeneratorConfig, generate_dataset


def _small_config() -> GeneratorConfig:
    return GeneratorConfig(
        n_accounts=500,
        n_base_transactions=2000,
        n_rings=5,
        ring_size_range=(4, 6),
        random_seed=42,
    )


def test_generate_dataset_shapes():
    result = generate_dataset(_small_config())
    assert len(result["accounts"]) == 500
    assert len(result["transactions"]) > 2000  # base + injected ring txns
    assert set(result["transactions"]["label"].unique()) <= {0, 1}


def test_injected_rings_are_labeled_fraud_and_share_an_entity():
    result = generate_dataset(_small_config())
    txns = result["transactions"]
    rings = result["rings_ground_truth"]

    assert len(rings) == 5
    for ring_id, member_accounts in rings.items():
        ring_txns = txns[txns["ring_id"] == ring_id]
        assert len(ring_txns) > 0
        assert (ring_txns["label"] == 1).all()
        assert set(ring_txns["account_id"]) <= set(member_accounts)
        # every ring's transactions share at least one device_id across all members
        assert ring_txns["device_id"].nunique() == 1


def test_entity_edges_recover_ring_membership():
    result = generate_dataset(_small_config())
    edges = result["entity_edges"]
    rings = result["rings_ground_truth"]

    assert not edges.empty
    linked_pairs = set(zip(edges["entity_a"], edges["entity_b"])) | set(
        zip(edges["entity_b"], edges["entity_a"])
    )
    # every ring of size >= 2 should contribute at least one recoverable edge
    for member_accounts in rings.values():
        if len(member_accounts) < 2:
            continue
        a, b = member_accounts[0], member_accounts[1]
        assert (a, b) in linked_pairs or (b, a) in linked_pairs
