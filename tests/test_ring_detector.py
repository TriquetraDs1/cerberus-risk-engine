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
    # The bar is deliberately lower than it used to be (it was 0.9, when every ring was a
    # clique on one shared device). Star, chain and partial rings are genuinely harder to
    # cluster — a `partial` ring contains members with no shared identifier at all, and no
    # graph method can recover those. Recovery near 1.0 on this generator would now
    # indicate a leak, not a good detector.
    assert report["mean_ring_recovery"] >= 0.5
    assert report["n_perfectly_recovered"] >= 3


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
    """Runs the detector the way it is actually deployed: with transactions, so the
    behavioural coordination filter applies. Structure alone cannot separate a four-person
    family from a four-person ring — they are the same graph — and testing the
    structure-only path here would assert a bound the deployed system does not rely on."""
    result = generate_dataset(_small_config())
    graph = build_graph(result["entity_edges"])
    detection = detect_communities(graph, transactions=result["transactions"])

    report = evaluate_against_ground_truth(
        detection, result["rings_ground_truth"], result["household_pairs"]
    )

    assert report["n_household_pairs"] > 0
    # not zero-or-one — a real number that could go into a slide, not a suspiciously
    # perfect result.
    assert 0.0 <= report["household_false_positive_rate"] <= 0.5


def test_behavioural_filter_beats_structure_alone_on_false_positives():
    """The point of the coordination layer, pinned as a test: with larger innocent
    households, structure alone false-positives badly, and adding the behavioural signal
    is what makes the detector usable. If a future change makes these equal, the
    coordination filter has stopped doing anything."""
    result = generate_dataset(_small_config())
    graph = build_graph(result["entity_edges"])

    structural = evaluate_against_ground_truth(
        detect_communities(graph), result["rings_ground_truth"], result["household_pairs"]
    )
    behavioural = evaluate_against_ground_truth(
        detect_communities(graph, transactions=result["transactions"]),
        result["rings_ground_truth"],
        result["household_pairs"],
    )
    assert (
        behavioural["household_false_positive_rate"] < structural["household_false_positive_rate"]
    )
