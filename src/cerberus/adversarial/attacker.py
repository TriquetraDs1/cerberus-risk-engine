"""The adaptive search: given an evasion strategy, find the parameters that minimize
detection by the CURRENT trained model + ring detector.

This is a randomized local search (multi-restart hill-climbing with an occasional
random jump to escape local optima) — not reinforcement learning, not a genetic
algorithm, deliberately. The detection landscape here is small and fairly smooth (more
structuring splits, more device rotation, or a slower ramp all monotonically push
detection down within their useful range), so a simple, fully explainable search finds
the same answer a heavier method would, without needing a library or a black-box
policy nobody on the panel can audit by reading the code.

Guardrail: every function in this module only ever scores *synthetic* rings against
*this repo's own* locally-loaded model artifacts. Nothing here sends a request
anywhere, targets a real system, or produces reusable offense tooling — see
strategies.py's module docstring and README.md's scope statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from cerberus.adversarial.strategies import EvasionRing
from cerberus.data.synthetic_rings import build_entity_edges
from cerberus.detection.ring_detector import build_graph, detect_communities
from cerberus.features.pipeline import FEATURE_COLUMNS, build_features


@dataclass
class DetectionScore:
    point_risk_caught_fraction: float  # fraction of the ring's txns routed block/review
    ring_recovered_fraction: float  # fraction of ring members co-located in one flagged community
    combined_score: float  # 0.5 * each — what the attacker minimizes, the defender maximizes


@dataclass
class SearchStep:
    iteration: int
    restart: int
    params: dict
    score: float


@dataclass
class SearchResult:
    strategy: str
    baseline_score: DetectionScore
    best_params: dict
    best_score: DetectionScore
    trace: list[SearchStep] = field(default_factory=list)


def score_ring(
    ring: EvasionRing,
    booster: lgb.Booster,
    calibrator,
    segment_routing: dict,
    global_default_threshold: float,
    min_ring_size: int = 3,
) -> DetectionScore:
    """Score one sandboxed ring against the current detection stack: the point-risk
    model + per-segment routing, AND the Louvain ring detector on this ring's own
    self-contained entity graph.
    """
    txns = ring.transactions
    if len(txns) == 0:
        return DetectionScore(0.0, 0.0, 0.0)

    entity_edges = build_entity_edges(txns)
    features = build_features(txns, entity_edges)
    X = features[FEATURE_COLUMNS]

    raw_scores = booster.predict(X)
    calibrated = calibrator.predict(raw_scores)

    decisions = []
    for i, seg in enumerate(features["segment"]):
        routing = segment_routing.get(seg)
        block_t = routing["block_threshold"] if routing else global_default_threshold
        review_t = routing["review_threshold"] if routing else block_t * 0.5
        decisions.append(calibrated[i] >= review_t)  # "caught" = routed to review or block
    point_caught_fraction = float(np.mean(decisions)) if decisions else 0.0

    graph = build_graph(entity_edges)
    if graph.number_of_nodes() == 0:
        ring_recovered_fraction = 0.0
    else:
        detection = detect_communities(graph, min_ring_size=min_ring_size)
        communities = [detection.community_of(m) for m in ring.member_account_ids]
        communities = [c for c in communities if c is not None]
        if not communities:
            ring_recovered_fraction = 0.0
        else:
            best = max(set(communities), key=communities.count)
            flagged = best in detection.flagged_ring_ids
            ring_recovered_fraction = (communities.count(best) / len(ring.member_account_ids)) if flagged else 0.0

    combined = 0.5 * point_caught_fraction + 0.5 * ring_recovered_fraction
    return DetectionScore(point_caught_fraction, ring_recovered_fraction, combined)


def adaptive_search(
    strategy_name: str,
    applier,
    param_bounds: dict[str, tuple[float, float]],
    make_base_ring,
    score_fn,
    rng: np.random.Generator,
    n_restarts: int = 5,
    n_steps: int = 15,
    random_jump_prob: float = 0.1,
) -> SearchResult:
    """Multi-restart hill-climbing: each restart draws a fresh sandbox ring instance,
    then locally searches that strategy's parameter space, perturbing one parameter at
    a time and accepting improvements (or, with `random_jump_prob`, a worse candidate —
    a small escape hatch against getting stuck in a local minimum).
    """
    param_names = list(param_bounds.keys())
    trace: list[SearchStep] = []

    baseline_ring = make_base_ring()
    baseline_score = score_fn(baseline_ring)

    best_params: dict | None = None
    best_score = DetectionScore(1.0, 1.0, 1.0)

    for restart in range(n_restarts):
        base_ring = make_base_ring()
        params = {name: float(rng.uniform(*param_bounds[name])) for name in param_names}
        current_ring = applier(base_ring, rng, **params)
        current_score = score_fn(current_ring)

        for step in range(n_steps):
            name = rng.choice(param_names)
            low, high = param_bounds[name]
            step_size = (high - low) * 0.2
            candidate = dict(params)
            candidate[name] = float(np.clip(params[name] + rng.normal(0, step_size), low, high))

            candidate_ring = applier(base_ring, rng, **candidate)
            candidate_score = score_fn(candidate_ring)
            trace.append(SearchStep(step, restart, dict(candidate), candidate_score.combined_score))

            if candidate_score.combined_score <= current_score.combined_score or rng.random() < random_jump_prob:
                params, current_score = candidate, candidate_score

            if current_score.combined_score < best_score.combined_score:
                best_score, best_params = current_score, dict(params)

    return SearchResult(
        strategy=strategy_name,
        baseline_score=baseline_score,
        best_params=best_params or {},
        best_score=best_score,
        trace=trace,
    )
