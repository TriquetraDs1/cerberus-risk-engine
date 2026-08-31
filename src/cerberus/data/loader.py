"""Dataset loading.

The synthetic generator (`synthetic_rings.py`) is the primary data source for this
project — it's the only source with fraud-ring ground truth, which the Day 3 graph layer
needs. A public Kaggle dataset (`creditcard.csv`) is optional; when present it is used to
**calibrate the generator's marginal distributions** — base fraud rate and the lognormal
amount parameters for legitimate vs. fraudulent transactions — rather than leaving them
as hand-picked constants.

The distinction that matters, and that should be stated exactly this way: marginals are
borrowed from real data, structure remains synthetic. The Kaggle set is anonymised PCA
components with no device / card / IP identifiers, so it cannot supply entity links or
ring ground truth at any price. Calibrating it does not make the data real; it removes
"the fraud rate is a number the author chose" from the list of things a reviewer has to
take on faith.

Drop a Kaggle creditcard.csv (https://www.kaggle.com/mlg-ulb/creditcardfraud) into
data/raw/ to enable it; the pipeline runs fully without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cerberus.common.config import KAGGLE_CREDITCARD_CSV


def _lognormal_params(amounts: pd.Series) -> tuple[float, float]:
    # Zero-amount rows exist in this dataset; log(0) is undefined, so drop them rather
    # than letting -inf poison the fit.
    positive = amounts[amounts > 0]
    if positive.empty:
        return 0.0, 1.0
    logs = np.log(positive)
    return float(logs.mean()), float(logs.std() or 1.0)


def kaggle_reference_stats() -> dict | None:
    """Summary of the optional Kaggle reference dataset, or None if it isn't present.

    Includes lognormal parameters for the legitimate and fraudulent amount
    distributions — what `calibrate_config_to_reference` needs to anchor the generator.
    """
    if not KAGGLE_CREDITCARD_CSV.exists():
        return None
    df = pd.read_csv(KAGGLE_CREDITCARD_CSV, usecols=["Amount", "Class"])

    legit_mu, legit_sigma = _lognormal_params(df.loc[df["Class"] == 0, "Amount"])
    fraud_mu, fraud_sigma = _lognormal_params(df.loc[df["Class"] == 1, "Amount"])

    return {
        "n_transactions": len(df),
        "fraud_rate": float(df["Class"].mean()),
        "median_amount": float(df["Amount"].median()),
        "legit_amount_lognormal": {"mu": legit_mu, "sigma": legit_sigma},
        "fraud_amount_lognormal": {"mu": fraud_mu, "sigma": fraud_sigma},
    }


def calibrate_config_to_reference(cfg, stats: dict | None = None, *, calibrate_amounts: bool = False):
    """Return a copy of `cfg` with its base fraud rate taken from the real reference
    dataset. No-op (returns `cfg` unchanged) when the file is absent, so the pipeline
    stays runnable without a Kaggle account.

    **The fraud rate transfers; the amount distribution does not.** How often fraud
    occurs is a dimensionless rate, comparable across markets, and 0.17% measured beats
    1.8% assumed. Amounts are denominated: the reference set is European card data with
    a ~$22 median, while this generator models an INR payment stream whose structuring
    threshold is ₹2000. Importing those amount parameters produced rings priced ~100x the
    legitimate median — internally incoherent data that collapsed the per-segment
    decision layer's savings to zero, because segments stopped differing in any way the
    cost matrix could exploit. That was measured, not assumed; see
    docs/EXPERIMENT_ADVANCED_TRAINING.md.

    `calibrate_amounts=True` opts into the amount transfer anyway. It is off by default
    and should stay off unless the reference dataset and the generator share a currency
    and market.
    """
    from dataclasses import replace

    stats = stats if stats is not None else kaggle_reference_stats()
    if not stats:
        return cfg

    updates = {"base_fraud_rate": stats["fraud_rate"], "calibrated_from_reference": True}
    if calibrate_amounts:
        legit, fraud = stats["legit_amount_lognormal"], stats["fraud_amount_lognormal"]
        updates.update(
            legit_amount_mu=legit["mu"],
            legit_amount_sigma=legit["sigma"],
            fraud_amount_mu=fraud["mu"],
            fraud_amount_sigma=fraud["sigma"],
        )
    return replace(cfg, **updates)
