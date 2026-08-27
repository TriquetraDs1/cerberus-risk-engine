"""Dataset loading.

The synthetic generator (`synthetic_rings.py`) is the primary data source for this
project — it's the only source with fraud-ring ground truth, which the Day 3 graph layer
needs. A public Kaggle dataset (e.g. `creditcard.csv`) is optional and, if present, is
used only to sanity-check that the synthetic base fraud rate and feature shapes are in a
realistic ballpark — it is not merged feature-for-feature into the synthetic set, since
its feature space (anonymized PCA components) doesn't carry entity identifiers anyway.

Drop a Kaggle creditcard.csv (https://www.kaggle.com/mlg-ulb/creditcardfraud) into
data/raw/ if you want that sanity check; the pipeline runs fully without it.
"""

from __future__ import annotations

import pandas as pd

from cerberus.common.config import KAGGLE_CREDITCARD_CSV


def kaggle_reference_stats() -> dict | None:
    """Return a small summary of the optional Kaggle reference dataset, if present."""
    if not KAGGLE_CREDITCARD_CSV.exists():
        return None
    df = pd.read_csv(KAGGLE_CREDITCARD_CSV, usecols=["Amount", "Class"])
    return {
        "n_transactions": len(df),
        "fraud_rate": float(df["Class"].mean()),
        "median_amount": float(df["Amount"].median()),
    }
