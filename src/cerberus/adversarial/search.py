"""Search strategies for the adversarial harness — how the attacker explores an evasion
strategy's parameter space.

Two implementations behind one signature:

  * `"hillclimb"` — the original multi-restart local search (`attacker.adaptive_search`).
    Fully auditable by reading forty lines, no dependency, and honest about being a
    local method that can miss a better evasion.
  * `"bayesopt"`  — Gaussian-process Bayesian optimisation (scikit-optimize). Models the
    detection surface and probes where evasion is most likely, so it finds deeper
    evasions on the same budget.

Why add the second: the harness's whole claim is "here is how far recall falls under
attack." That number is only as trustworthy as the attacker producing it. A weak search
reports a flattering decay figure — which is the failure mode this project exists to
argue against. A stronger searcher makes the reported robustness *lower* and *more
honest*, which is the right direction for a claim to move.

Both share the sandbox guardrail unchanged: fresh synthetic rings, scored only against
locally-loaded model artifacts. Swapping the optimiser changes which parameters get
tried, nothing about what is attacked.
"""

from __future__ import annotations

import numpy as np

from cerberus.adversarial.attacker import DetectionScore, SearchResult, SearchStep, adaptive_search

SEARCHERS = ("hillclimb", "bayesopt")


def bayesopt_search(
    strategy_name: str,
    applier,
    param_bounds: dict[str, tuple[float, float]],
    make_base_ring,
    score_fn,
    rng: np.random.Generator,
    n_restarts: int = 5,
    n_steps: int = 15,
) -> SearchResult:
    """Gaussian-process Bayesian optimisation over the same parameter space.

    Budget is matched to the hill-climb's (`n_restarts * n_steps` evaluations) so the two
    are comparable: any difference in the evasion found is the optimiser's doing, not a
    larger allowance.

    A fresh sandbox ring per evaluation would make the objective stochastic and defeat
    the surrogate model, so one base ring is drawn per restart — the same convention the
    hill-climb uses within a restart.
    """
    from skopt import gp_minimize
    from skopt.space import Real

    param_names = list(param_bounds.keys())
    space = [Real(*param_bounds[name], name=name) for name in param_names]
    trace: list[SearchStep] = []

    baseline_score = score_fn(make_base_ring())

    best_params: dict | None = None
    best_score = DetectionScore(1.0, 1.0, 1.0)

    # Split the budget across restarts, mirroring the hill-climb's structure. gp_minimize
    # needs a handful of random points before the surrogate is worth anything, so keep a
    # floor of ~10 calls per restart.
    n_calls = max(10, n_steps)

    for restart in range(n_restarts):
        base_ring = make_base_ring()
        step_counter = {"i": 0}

        def objective(values, _ring=base_ring, _restart=restart, _counter=step_counter):
            params = dict(zip(param_names, (float(v) for v in values), strict=True))
            score = score_fn(applier(_ring, rng, **params))
            trace.append(SearchStep(_counter["i"], _restart, dict(params), score.combined_score))
            _counter["i"] += 1
            return score.combined_score

        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=min(5, n_calls - 1),
            # Derive each restart's seed from the harness rng so the whole run stays
            # reproducible under CERBERUS_RANDOM_SEED.
            random_state=int(rng.integers(0, 2**31 - 1)),
            verbose=False,
        )

        params = dict(zip(param_names, (float(v) for v in result.x), strict=True))
        # gp_minimize returns the objective value, not the DetectionScore behind it —
        # re-score the winner so the breakdown (point-risk vs. ring) is real, not
        # reconstructed from the scalar.
        score = score_fn(applier(base_ring, rng, **params))
        if score.combined_score < best_score.combined_score:
            best_score, best_params = score, params

    return SearchResult(
        strategy=strategy_name,
        baseline_score=baseline_score,
        best_params=best_params or {},
        best_score=best_score,
        trace=trace,
    )


def get_searcher(name: str):
    """Return the search function for `name`. Both share `adaptive_search`'s signature."""
    if name == "hillclimb":
        return adaptive_search
    if name == "bayesopt":
        return bayesopt_search
    raise ValueError(f"Unknown searcher {name!r} — expected one of {SEARCHERS}.")
