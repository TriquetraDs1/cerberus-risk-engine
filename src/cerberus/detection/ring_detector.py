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
    # community_id -> behavioural coordination score in [0, 1]. Empty when the detector
    # was run on structure alone (no transactions supplied).
    coordination: dict[int, float] = field(default_factory=dict)

    def community_of(self, account_id: str) -> int | None:
        return self.partition.get(account_id)


# A community must clear this to be called a ring.
#
# Chosen, not fitted, and the whole trade-off curve is written into
# reports/ring_detection_report.json so the choice is inspectable rather than asserted.
# Measured on the current generator (ring communities score 0.23-0.41, innocent ones
# 0.10-0.29, so the classes overlap and no threshold is free):
#
#   0.26 -> catches 88% of ring communities, flags 28% of innocent ones
#   0.28 -> catches 71%, flags 17%          <- default
#   0.30 -> catches 59%, flags 0%
#
# 0.28 sits between the two flattering ends. Pushing to 0.30 would let the report claim a
# 0% false-positive rate while silently missing four rings in ten, which is precisely the
# kind of number this project exists to argue against. A deployment with a real cost model
# for analyst review time should derive this rather than inherit it.
COORDINATION_THRESHOLD = 0.28

# Transactions this close together count as burst-concurrent.
BURST_WINDOW = pd.Timedelta(hours=6)


def coordination_score(members: list[str], transactions: pd.DataFrame) -> float:
    """How coordinated a community's spending looks, in [0, 1].

    This exists because **structure alone cannot separate a fraud ring from a family.**
    Four people sharing one phone produce the same graph whether they are relatives or
    colluding: same size, same density, same degree. On a dataset where innocent
    households are pairs, community size hides that; on one where households have three
    to five members, Louvain on structure alone false-positives on almost every family
    (94% measured — see docs/EXPERIMENT_ADVANCED_TRAINING.md).

    What differs is not the graph, it is the behaviour. A ring transacts *together*: a
    burst inside a few hours, amounts clustered just under a reporting threshold, most
    members active at once. A household transacts independently across weeks at whatever
    amounts daily life produces.

    Three signals, averaged:
      * **burst concentration** — the largest fraction of the community's transactions
        falling inside one 6-hour window.
      * **amount clustering** — how tightly amounts group, via a normalised inverse
        coefficient of variation. Structured amounts are near-identical; real spending is
        spread.
      * **member synchrony** — the fraction of members active in that same busiest window.
        One member's spree is not a ring; five members in the same six hours is.

    Returns 0.0 when there is nothing to judge (a community with no transactions), which
    keeps an absent-data community unflagged rather than flagged by default.
    """
    rows = transactions[transactions["account_id"].isin(members)]
    if len(rows) < 2:
        return 0.0

    times = pd.to_datetime(rows["timestamp"]).sort_values()
    amounts = rows["amount"].to_numpy(dtype=float)

    # Busiest 6-hour window, found by sliding the window start over each transaction.
    best_count, best_window_members = 0, 0
    for start in times:
        window = rows[
            (pd.to_datetime(rows["timestamp"]) >= start)
            & (pd.to_datetime(rows["timestamp"]) < start + BURST_WINDOW)
        ]
        if len(window) > best_count:
            best_count = len(window)
            best_window_members = window["account_id"].nunique()

    burst_concentration = best_count / len(rows)
    synchrony = best_window_members / max(len(members), 1)

    mean_amount = float(amounts.mean())
    cv = float(amounts.std() / mean_amount) if mean_amount > 0 else 1.0
    # cv near 0 means near-identical amounts (structuring); cv >= 1 is ordinary spread.
    amount_clustering = max(0.0, 1.0 - min(cv, 1.0))

    return float((burst_concentration + synchrony + amount_clustering) / 3.0)


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
    graph: nx.Graph,
    resolution: float | None = None,
    min_ring_size: int = 3,
    transactions: pd.DataFrame | None = None,
    coordination_threshold: float = COORDINATION_THRESHOLD,
) -> RingDetectionResult:
    """Louvain community detection, then a behavioural check before anything is called a
    ring.

    **Structure proposes, behaviour disposes.** Louvain finds groups of accounts wired
    together by shared devices and cards; that is a candidate list, not a verdict, because
    a family and a fraud ring produce the same wiring. When `transactions` is supplied, a
    candidate is only flagged if it *also* looks coordinated — see `coordination_score`.

    Passing no transactions falls back to the structure-only rule (size >= min_ring_size).
    That path is kept because the adversarial harness scores self-contained sandbox rings
    where the behavioural signal is a property of the attack being tested rather than
    something to re-derive, and because it is what the earlier reports were produced with.
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

    candidates = [cid for cid, members in communities.items() if len(members) >= min_ring_size]

    coordination: dict[int, float] = {}
    if transactions is not None and len(transactions) > 0:
        for cid in candidates:
            coordination[cid] = coordination_score(communities[cid], transactions)
        flagged = [cid for cid in candidates if coordination[cid] >= coordination_threshold]
    else:
        flagged = candidates

    return RingDetectionResult(
        graph=graph,
        partition=partition,
        communities=dict(communities),
        flagged_ring_ids=flagged,
        coordination=coordination,
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
