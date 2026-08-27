"""Day 3 sanity tests: does Louvain actually recover injected rings, and does it avoid
over-flagging innocent entity-sharing as coordinated fraud?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.data.synthetic_rings import GeneratorConfig, generate_dataset
from cerberus.detection.ring_detector import (
    build_graph,
    detect_communities,
    evaluate_against_ground_truth,
)


def _small_config() -> GeneratorConfig:
    return GeneratorConfig(
        n_accounts=800,
        n_base_transactions=3000,
        n_rings=8,
        ring_size_range=(4, 7),
        household_sharing_rate=0.05,
        random_seed=7,
    )


def test_louvain_recovers_injected_rings():
    result = generate_dataset(_small_config())
    graph = build_graph(result["entity_edges"])
    detection = detect_communities(graph)

    report = evaluate_against_ground_truth(
        detection, result["rings_ground_truth"], result["household_pairs"]
    )

    assert report["n_rings"] == 8
    # every ring should land almost entirely in one community; the exact recovery
    # rate can vary slightly with the graph structure, so assert a high bar, not 100%.
    assert report["mean_ring_recovery"] >= 0.9
    assert report["n_perfectly_recovered"] >= 6


def test_isolated_pair_alone_is_never_flagged():
    """A bare two-account edge (no third party) can never be a size->=3 flagged
    community on its own — this is the honest-by-construction guardrail against
    treating every shared device as a ring.
    """
    result = generate_dataset(_small_config())
    graph = build_graph(result["entity_edges"])
    detection = detect_communities(graph, min_ring_size=3)

    for community_id, members in detection.communities.items():
        if len(members) < 3:
            assert community_id not in detection.flagged_ring_ids


def test_household_false_positive_rate_is_reported_and_bounded():
    result = generate_dataset(_small_config())
    graph = build_graph(result["entity_edges"])
    detection = detect_communities(graph)

    report = evaluate_against_ground_truth(
        detection, result["rings_ground_truth"], result["household_pairs"]
    )

    assert report["n_household_pairs"] > 0
    # not zero-or-one — a real number that could go into a slide, not a suspiciously
    # perfect result.
    assert 0.0 <= report["household_false_positive_rate"] <= 0.5
