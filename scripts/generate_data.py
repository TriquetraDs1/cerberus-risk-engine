#!/usr/bin/env python
"""Day 1-2: generate the synthetic transaction set with injectable fraud rings.

Usage:
    python scripts/generate_data.py                    # assumed 1.8% base fraud rate
    python scripts/generate_data.py --calibrate-rate   # real 0.17%, from data/raw/creditcard.csv

Why the real rate is opt-in rather than the default, stated plainly because it looks
like the wrong way round: 0.17% is the truthful number and the model scores *better*
on it (ROC-AUC 0.858 vs 0.815, PR-AUC 0.509 vs 0.314). But at a tenth the fraud, a
60k-transaction dataset holds ~100 base-fraud rows, and three downstream things thin
out with them — the per-segment decision layer's advantage (16.5% -> 9.0%), the review
tier (it empties), and the adversarial harness's unattacked baselines. Those are
small-sample artefacts, not findings about the method.

The statistically correct fix is more transactions, not a fatter fraud rate; that
costs pipeline time this build hasn't spent yet. Until then both configurations are
reproducible from this flag and both sets of numbers are in
docs/EXPERIMENT_ADVANCED_TRAINING.md. Cite whichever you run — just say which.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.common.config import (
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_HOUSEHOLD_PAIRS_JSON,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.data.loader import calibrate_config_to_reference, kaggle_reference_stats
from cerberus.data.synthetic_rings import GeneratorConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--calibrate-rate",
        action="store_true",
        help="Take the base fraud rate from data/raw/creditcard.csv instead of the assumed default.",
    )
    args = parser.parse_args()

    defaults = GeneratorConfig()
    cfg = defaults
    kaggle_stats = kaggle_reference_stats()

    if args.calibrate_rate and kaggle_stats:
        cfg = calibrate_config_to_reference(defaults, kaggle_stats)
        print(
            f"Calibrating against data/raw/creditcard.csv "
            f"({kaggle_stats['n_transactions']:,} real transactions):"
        )
        print(
            f"  base fraud rate: {defaults.base_fraud_rate:.4f} (assumed) "
            f"-> {cfg.base_fraud_rate:.4f} (measured)"
        )
        print(
            f"  amounts left synthetic: the reference is ~${kaggle_stats['median_amount']:.0f}-median "
            "card data, this generator models an INR stream. The rate transfers, the currency doesn't."
        )
        print("  entity structure, rings, and timing stay synthetic — they have to: the")
        print("  reference set is anonymised PCA components with no entity identifiers.\n")
    elif args.calibrate_rate:
        print("--calibrate-rate given but data/raw/creditcard.csv is missing; using assumed defaults.\n")
    elif kaggle_stats:
        print(
            f"Reference data present. Measured fraud rate is {kaggle_stats['fraud_rate']:.4f} vs. the "
            f"{defaults.base_fraud_rate:.4f} assumed here — pass --calibrate-rate to use it.\n"
        )

    result = generate_dataset(cfg)

    txns = result["transactions"]
    edges = result["entity_edges"]
    rings = result["rings_ground_truth"]
    household_pairs = result["household_pairs"]

    txns.to_csv(SYNTHETIC_TRANSACTIONS_CSV, index=False)
    edges.to_csv(SYNTHETIC_ENTITY_EDGES_CSV, index=False)
    SYNTHETIC_RINGS_JSON.write_text(json.dumps(rings, indent=2))
    SYNTHETIC_HOUSEHOLD_PAIRS_JSON.write_text(json.dumps(household_pairs, indent=2))

    n_ring_txns = txns["ring_id"].notna().sum()
    print(f"Generated {len(txns):,} transactions ({txns['label'].mean():.2%} labeled fraud)")
    print(f"  base fraud: {len(txns) - n_ring_txns:,}  |  ring-injected fraud: {n_ring_txns:,}")
    print(f"  {len(rings)} injected rings, {len(edges):,} entity-link edges")
    print(f"  {len(household_pairs)} innocent household-sharing pairs (FP validation set)")
    print(f"  wrote {SYNTHETIC_TRANSACTIONS_CSV}")
    print(f"  wrote {SYNTHETIC_ENTITY_EDGES_CSV}")
    print(f"  wrote {SYNTHETIC_RINGS_JSON}")
    print(f"  wrote {SYNTHETIC_HOUSEHOLD_PAIRS_JSON}")

    print(
        f"  marginals: {'calibrated from real data' if cfg.calibrated_from_reference else 'assumed defaults'}"
    )


if __name__ == "__main__":
    main()
