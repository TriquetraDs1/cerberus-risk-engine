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
        # Rings are no longer uniform cliques — they come in clique / star / chain /
        # partial shapes, so "every member on one device" is deliberately false now.
        # What must still hold is that a ring is *linked*: at least two of its members
        # share a device. A ring with no shared identifier at all would be invisible to
        # the graph layer by construction and could never be recovered, which would make
        # the recovery metric meaningless rather than hard.
        device_counts = ring_txns.groupby("device_id")["account_id"].nunique()
        assert (device_counts >= 2).any(), f"{ring_id} has no device shared by two members"


def test_entity_edges_recover_ring_membership():
    """Every ring must leave a recoverable trace in the entity graph — but not the same
    trace it used to.

    The old assertion was that the ring's first two members share an edge, which only held
    because every ring was a clique on one device. Rings now come in four shapes: a star's
    leaves connect to the hub and not to each other, a chain connects consecutive members
    only, and a `partial` ring leaves some members with no shared identifier at all.

    So the contract is weaker and more honest: at least two members of each ring are
    connected, so the ring is *reachable* by a graph method. Anything stronger would be
    asserting the clique topology this generator was changed to stop producing; anything
    weaker would let a completely unlinked ring through and make ring recovery
    unmeasurable rather than merely hard.
    """
    result = generate_dataset(_small_config())
    edges = result["entity_edges"]
    rings = result["rings_ground_truth"]

    assert not edges.empty
    linked_pairs = set(zip(edges["entity_a"], edges["entity_b"], strict=True)) | set(
        zip(edges["entity_b"], edges["entity_a"], strict=True)
    )

    for ring_id, member_accounts in rings.items():
        if len(member_accounts) < 2:
            continue
        members = set(member_accounts)
        has_internal_edge = any(
            (a, b) in linked_pairs for a in members for b in members if a != b
        )
        assert has_internal_edge, f"{ring_id} left no recoverable edge in the entity graph"
