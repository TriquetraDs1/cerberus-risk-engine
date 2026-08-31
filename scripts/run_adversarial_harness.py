#!/usr/bin/env python
"""Day 5-6: the adversarial hardening harness.

Builds a detector (already trained), attacks it with an adaptive search over three
evasion strategies, measures the damage (recall decay), retrains on what the search
found, and re-attacks the hardened model with a *fresh* search to prove the recovery
is real — not just "we memorized this one attack."

Defensive-only: every attack here scores synthetic sandbox rings against this repo's
own local model files. See README.md's scope statement and the module docstrings in
src/cerberus/adversarial/ for the guardrail spelled out in full.

Usage:
    python scripts/generate_data.py
    python scripts/detect_rings.py
    python scripts/train_baseline.py
    python scripts/build_decision_layer.py
    python scripts/run_adversarial_harness.py [--min-recovery -0.05]

`--min-recovery` is a CI regression gate, not a quality bar: it fails the build only if
hardening makes a strategy's detection *worse* than before hardening (a real bug — bad
data, an overfit retrain, a broken pipeline), not if a strategy's recovery is honestly
small. The identity-rotation strategy's near-zero ring-detector recovery is a known,
accepted, reported limitation (see docs/ARCHITECTURE.md) — the gate must not fail the
build over an honestly-reported number, only over the number getting worse over time.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cerberus.adversarial.harness import (
    generate_hardening_examples,
    report_to_dict,
    retrain_hardened_model,
    run_evasion_search,
    score_summary,
)
from cerberus.adversarial.search import SEARCHERS
from cerberus.common.config import (
    BASELINE_MODEL_PATH,
    CALIBRATOR_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
    settings,
)
from cerberus.detection.point_risk import three_way_split
from cerberus.features.pipeline import build_features

DECISION_LAYER_JSON = REPORTS_DIR / "decision_layer.json"
ADVERSARIAL_REPORT_JSON = REPORTS_DIR / "adversarial_hardening_report.json"
RECALL_DECAY_FIGURE = FIGURES_DIR / "recall_decay.png"

N_RESTARTS = 5
N_STEPS = 15


def plot_recall_decay(reports: dict) -> None:
    strategies = list(reports.keys())
    baseline = [reports[s]["baseline_detection"]["combined_score"] for s in strategies]
    evaded = [reports[s]["evaded_original_model"]["combined_score"] for s in strategies]
    hardened = [reports[s]["evaded_hardened_model"]["combined_score"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.bar(x - width, baseline, width, label="Unattacked", color="#059669")
    ax.bar(x, evaded, width, label="Under attack (original model)", color="#e11d48")
    ax.bar(x + width, hardened, width, label="Under attack (hardened model)", color="#2563eb")

    ax.set_ylabel("Detection score (0 = fully evaded, 1 = fully caught)")
    ax.set_title("Adversarial hardening: detection before / under attack / after hardening")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in strategies])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(RECALL_DECAY_FIGURE)
    print(f"Wrote {RECALL_DECAY_FIGURE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-recovery",
        type=float,
        default=None,
        help=(
            "CI regression gate: exit non-zero if any strategy's "
            "recall_recovered_after_hardening falls below this value. Not a quality "
            "bar — see the module docstring for why this must stay a regression check, "
            "not a bar over an honestly-reported small recovery."
        ),
    )
    parser.add_argument(
        "--searcher",
        choices=SEARCHERS,
        default="hillclimb",
        help=(
            "How the attacker explores each strategy's parameter space. 'hillclimb' is "
            "the dependency-free local search; 'bayesopt' models the detection surface "
            "and finds deeper evasions on the same evaluation budget (needs the 'tuning' "
            "extra). A stronger searcher reports *lower*, more honest robustness — see "
            "cerberus.adversarial.search."
        ),
    )
    args = parser.parse_args()

    for path in (SYNTHETIC_TRANSACTIONS_CSV, BASELINE_MODEL_PATH, CALIBRATOR_PATH, DECISION_LAYER_JSON):
        if not path.exists():
            raise SystemExit(f"Missing {path} — run the earlier pipeline scripts first.")

    rng = np.random.default_rng(settings.random_seed)

    print("Loading the trained model, calibrator, and decision layer...")
    booster = lgb.Booster(model_file=str(BASELINE_MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)
    decision_layer = json.loads(DECISION_LAYER_JSON.read_text())
    segment_routing = decision_layer["segments"]
    global_default_threshold = decision_layer["global_default_threshold"]

    print("Rebuilding the original train/calib split (same seed, deterministic)...")
    txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, parse_dates=["timestamp"])
    edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
    features = build_features(txns, edges)
    train_df, calib_df, _ = three_way_split(features)

    print(
        f"\n--- Phase 1: attacking the original model ({N_RESTARTS} restarts x {N_STEPS} "
        f"steps per strategy, searcher={args.searcher}) ---"
    )
    original_results = run_evasion_search(
        booster, calibrator, segment_routing, global_default_threshold, rng, N_RESTARTS, N_STEPS,
        searcher=args.searcher,
    )
    for name, result in original_results.items():
        print(
            f"  {name:20s} baseline={result.baseline_score.combined_score:.2f}  "
            f"best evasion={result.best_score.combined_score:.2f}  params={result.best_params}"
        )

    print("\n--- Phase 2: hardening — retraining on the attacks the search found ---")
    adversarial_examples = generate_hardening_examples(original_results, rng)
    print(f"  generated {len(adversarial_examples)} adversarial training examples")
    hardened_model, hardened_calibrator = retrain_hardened_model(
        train_df, calib_df, adversarial_examples, settings.random_seed
    )

    print("\n--- Phase 3: re-attacking the hardened model (fresh search, same budget) ---")
    hardened_results = run_evasion_search(
        hardened_model, hardened_calibrator, segment_routing, global_default_threshold, rng,
        N_RESTARTS, N_STEPS, searcher=args.searcher,
    )
    for name, result in hardened_results.items():
        print(f"  {name:20s} best evasion vs. hardened model={result.best_score.combined_score:.2f}")

    reports = score_summary(original_results)
    for report in reports:
        report.evaded_hardened = hardened_results[report.strategy].best_score

    report_dict = report_to_dict(reports)

    print("\n--- Honest recall-decay report ---")
    for strategy, r in report_dict.items():
        print(
            f"  {strategy:20s} decay under attack: {r['recall_decay_original']:+.2f}   "
            f"recovered after hardening: {r['recall_recovered_after_hardening']:+.2f}"
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_restarts": N_RESTARTS,
        "n_steps": N_STEPS,
        "searcher": args.searcher,
        "n_adversarial_examples": len(adversarial_examples),
        "strategies": report_dict,
        "limitations": [
            "This validates robustness to the three evasion classes implemented here, "
            "searched adaptively within their parameter ranges — not a general "
            "adversarial-robustness guarantee. A real fraud ring is not obligated to "
            "restrict itself to these strategies.",
            "The search is not a certified worst-case search — it can miss a better "
            "evasion than the one it reports. Which optimiser produced these numbers is "
            "recorded in the `searcher` field above; 'bayesopt' explores the surface "
            "more thoroughly than the default 'hillclimb'.",
            "The ring detector is not retrained during hardening (Louvain is "
            "unsupervised); identity rotation's evasion of the graph layer is reported "
            "honestly as an open vulnerability, not silently patched.",
        ],
    }
    ADVERSARIAL_REPORT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {ADVERSARIAL_REPORT_JSON}")

    plot_recall_decay(report_dict)

    # Save the hardened artifacts alongside the originals — never overwrite the
    # baseline, so a reviewer can diff "before" and "after" directly.
    from cerberus.common.config import MODELS_DIR

    hardened_model.booster_.save_model(str(MODELS_DIR / "point_risk_hardened.txt"))
    joblib.dump(hardened_calibrator, str(MODELS_DIR / "point_risk_calibrator_hardened.joblib"))
    print(f"Wrote {MODELS_DIR / 'point_risk_hardened.txt'}")

    if args.min_recovery is not None:
        regressions = {
            strategy: r["recall_recovered_after_hardening"]
            for strategy, r in report_dict.items()
            if r["recall_recovered_after_hardening"] < args.min_recovery
        }
        if regressions:
            print(
                f"\nREGRESSION GATE FAILED (--min-recovery {args.min_recovery}): "
                f"{regressions}"
            )
            raise SystemExit(1)
        print(f"\nRegression gate passed: every strategy recovered >= {args.min_recovery} after hardening.")


if __name__ == "__main__":
    main()
