#!/usr/bin/env python
"""Day 1-2: generate the synthetic transaction set with injectable fraud rings.

Usage:
    python scripts/generate_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.common.config import (
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.data.loader import kaggle_reference_stats
from cerberus.data.synthetic_rings import GeneratorConfig, generate_dataset


def main() -> None:
    cfg = GeneratorConfig()
    result = generate_dataset(cfg)

    txns = result["transactions"]
    edges = result["entity_edges"]
    rings = result["rings_ground_truth"]

    txns.to_csv(SYNTHETIC_TRANSACTIONS_CSV, index=False)
    edges.to_csv(SYNTHETIC_ENTITY_EDGES_CSV, index=False)
    SYNTHETIC_RINGS_JSON.write_text(json.dumps(rings, indent=2))

    n_ring_txns = txns["ring_id"].notna().sum()
    print(f"Generated {len(txns):,} transactions ({txns['label'].mean():.2%} labeled fraud)")
    print(f"  base fraud: {len(txns) - n_ring_txns:,}  |  ring-injected fraud: {n_ring_txns:,}")
    print(f"  {len(rings)} injected rings, {len(edges):,} entity-link edges")
    print(f"  wrote {SYNTHETIC_TRANSACTIONS_CSV}")
    print(f"  wrote {SYNTHETIC_ENTITY_EDGES_CSV}")
    print(f"  wrote {SYNTHETIC_RINGS_JSON}")

    kaggle_stats = kaggle_reference_stats()
    if kaggle_stats:
        print(f"\nOptional Kaggle reference (data/raw/creditcard.csv) found:")
        print(f"  {kaggle_stats}")
    else:
        print(
            "\nNo Kaggle creditcard.csv found in data/raw/ — skipping reference sanity "
            "check (optional, see src/cerberus/data/loader.py)."
        )


if __name__ == "__main__":
    main()
