"""Day 3: ring detector.

Builds an entity-link graph on shared device/IP/card across accounts and runs Louvain
community detection to flag coordinated clusters — the second detection head described
in docs/ARCHITECTURE.md, run independently of the point-risk model so either can be
retrained without touching the other.

Two things are validated here, not just reported as a bare "it works":
  1. Recovery: does each *injected* fraud ring actually land inside one detected
     community? (recall of ring recovery, per docs/ARCHITECTURE.md panel-pushback #1)
  2. Honesty: does the detector avoid flagging *innocent* entity-sharing (e.g. a
     household sharing a device) as a coordinated ring? (the honest FP-cost story for
     the graph layer, per docs/ARCHITECTURE.md panel-pushback #2)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

try:
    import community as community_louvain  # python-louvain package, import name "community"
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "python-louvain is required for ring detection. Install with `pip install python-louvain`."
    ) from exc

from cerberus.common.config import settings


@dataclass
class RingDetectionResult:
    graph: nx.Graph
    partition: dict[str, int]  # account_id -> community_id
    communities: dict[int, list[str]]  # community_id -> [account_id, ...]
    flagged_ring_ids: list[int] = field(default_factory=list)  # communities treated as "rings"

    def community_of(self, account_id: str) -> int | None:
        return self.partition.get(account_id)


def build_graph(entity_edges: pd.DataFrame) -> nx.Graph:
    """Weighted undirected graph: nodes are accounts, edges are shared entities.
    Edge weight = number of distinct shared-entity links between the pair (an account
    pair sharing both a device AND a card is a stronger signal than sharing just one).
    """
    graph = nx.Graph()
    for _, row in entity_edges.iterrows():
        a, b, weight = row["entity_a"], row["entity_b"], row["weight"]
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += weight
        else:
            graph.add_edge(a, b, weight=weight)
    return graph


def detect_communities(
    graph: nx.Graph, resolution: float | None = None, min_ring_size: int = 3
) -> RingDetectionResult:
    """Run Louvain community detection and flag communities of size >= min_ring_size
    as candidate rings. Isolated pairs (size 2, e.g. a single shared device between two
    accounts) are NOT auto-flagged — that's exactly the innocent-household-sharing shape,
    and flagging every shared device as a "ring" would make the honest FP rate meaningless.
    """
    if graph.number_of_nodes() == 0:
        return RingDetectionResult(graph=graph, partition={}, communities={})

    resolution = resolution if resolution is not None else settings.louvain_resolution
    partition = community_louvain.best_partition(
        graph, weight="weight", resolution=resolution, random_state=settings.random_seed
    )

    communities: dict[int, list[str]] = defaultdict(list)
    for account_id, community_id in partition.items():
        communities[community_id].append(account_id)

    flagged = [cid for cid, members in communities.items() if len(members) >= min_ring_size]

    return RingDetectionResult(
        graph=graph, partition=partition, communities=dict(communities), flagged_ring_ids=flagged
    )


def evaluate_against_ground_truth(
    result: RingDetectionResult,
    rings_ground_truth: dict[str, list[str]],
    household_pairs: list[tuple[str, str]],
) -> dict:
    """Two honest numbers, not one vanity number:

    - `ring_recovery`: per injected ring, the fraction of its members that landed in the
      single detected community holding the most of that ring (1.0 = perfectly recovered).
    - `household_false_positive_rate`: fraction of innocent household pairs whose two
      accounts ended up in the SAME flagged (size >= min_ring_size) community — i.e.
      would have been wrongly escalated as a coordinated ring.
    """
    per_ring_recovery = {}
    for ring_id, members in rings_ground_truth.items():
        detected_communities = [result.community_of(m) for m in members if result.community_of(m) is not None]
        if not detected_communities:
            per_ring_recovery[ring_id] = 0.0
            continue
        # the community that captured the most of this ring's members
        best_community = max(set(detected_communities), key=detected_communities.count)
        recovered = detected_communities.count(best_community)
        per_ring_recovery[ring_id] = recovered / len(members)

    mean_recovery = (
        sum(per_ring_recovery.values()) / len(per_ring_recovery) if per_ring_recovery else 0.0
    )
    perfectly_recovered = sum(1 for v in per_ring_recovery.values() if v == 1.0)

    flagged_set = set(result.flagged_ring_ids)
    household_false_positives = 0
    for a, b in household_pairs:
        ca, cb = result.community_of(a), result.community_of(b)
        if ca is not None and ca == cb and ca in flagged_set:
            household_false_positives += 1
    household_fp_rate = (
        household_false_positives / len(household_pairs) if household_pairs else 0.0
    )

    return {
        "n_rings": len(rings_ground_truth),
        "n_perfectly_recovered": perfectly_recovered,
        "mean_ring_recovery": mean_recovery,
        "per_ring_recovery": per_ring_recovery,
        "n_household_pairs": len(household_pairs),
        "n_household_false_positives": household_false_positives,
        "household_false_positive_rate": household_fp_rate,
        "n_flagged_communities": len(result.flagged_ring_ids),
        "n_total_communities": len(result.communities),
    }
