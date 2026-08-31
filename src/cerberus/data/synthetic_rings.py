"""Synthetic transaction generator with injectable fraud rings.

Public fraud datasets (e.g. Kaggle's creditcard.csv) give a realistic base rate for
point-in-time fraud but have no labeled *fraud rings* — no ground truth for "these five
accounts are colluding." The graph layer (Day 3) needs that ground truth to be validated
against something, so this module manufactures it.

Everything here is synthetic and only ever used to train/evaluate this repo's own models.
No real accounts, devices, or payment instruments are involved. See README.md for the
project-wide defensive-use scope statement.

Two kinds of structure are injected:
  1. Base transactions with an independent, per-transaction fraud label (ordinary
     point-fraud: stolen card used once, out-of-pattern purchase, etc.) — this is what the
     point-risk model (Day 1-2) trains on.
  2. Injected fraud rings: small groups of accounts that share a device/card and transact
     in a coordinated burst with structured (just-under-threshold) amounts — this is what
     the ring detector (Day 3) is validated against.

A small fraction of *legitimate* account pairs are also given a shared device (e.g.
family members) with no fraud label, specifically so the ring detector's false-positive
rate against innocent entity-sharing can be measured honestly later (see
docs/ARCHITECTURE.md, "Louvain on synthetic data is easy").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cerberus.common.config import settings


def _short_id(prefix: str, rng: np.random.Generator) -> str:
    # rng is threaded through so callers stay deterministic under a fixed seed even
    # though the id bytes themselves come from uuid4 (numpy has no built-in UUID source).
    raw = rng.integers(0, 2**32 - 1, size=4, dtype=np.uint32).tobytes()  # 16 bytes
    return f"{prefix}_{uuid.UUID(bytes=raw).hex[:10]}"


# Merchant segments give the Day 4 decision layer something real to segment on: fraud
# economics genuinely differ by category (a lost travel booking costs far more than a
# lost grocery order; digital goods carry higher friendly-fraud/chargeback rates than
# physical retail), so treating every transaction with one global cost ratio is the
# thing a real payments team would immediately flag as naive. See
# cerberus.decision.cost_matrix for how these translate into per-segment thresholds.
SEGMENTS = ("grocery_essentials", "electronics_highvalue", "digital_subscription", "travel_luxury")
SEGMENT_WEIGHTS = (0.40, 0.25, 0.25, 0.10)
# (amount_multiplier, fraud_rate_multiplier) relative to the base distributions.
SEGMENT_PROFILES = {
    "grocery_essentials": (0.35, 0.6),
    "electronics_highvalue": (2.4, 1.1),
    "digital_subscription": (0.55, 1.8),  # friendly fraud / chargebacks skew this up
    "travel_luxury": (5.0, 1.3),
}


@dataclass
class GeneratorConfig:
    n_accounts: int = 5000
    n_base_transactions: int = 60_000
    base_fraud_rate: float = 0.018  # ordinary, uncoordinated fraud
    household_sharing_rate: float = 0.03  # fraction of accounts in an innocent shared-device pair

    # Lognormal amount parameters, legitimate vs. fraudulent. Defaults are hand-picked
    # for a plausible shape; `data.loader.calibrate_config_to_reference` replaces them
    # (and base_fraud_rate) with values measured off the real Kaggle dataset when
    # data/raw/creditcard.csv is present.
    legit_amount_mu: float = 5.2
    legit_amount_sigma: float = 0.9
    fraud_amount_mu: float = 6.5  # heavier tail, higher mean
    fraud_amount_sigma: float = 1.1
    # Set by the calibration helper so downstream reports can state, honestly, whether
    # the run's marginals were measured or assumed.
    calibrated_from_reference: bool = False

    n_rings: int = 25
    ring_size_range: tuple[int, int] = (4, 12)
    txns_per_ring_account_range: tuple[int, int] = (2, 5)
    ring_burst_minutes_range: tuple[int, int] = (15, 90)
    structuring_threshold: float = 2000.0  # e.g. an INR reporting/review threshold

    start_time: datetime = field(default_factory=lambda: datetime(2026, 1, 1))
    window_days: int = 60

    random_seed: int = settings.random_seed


def generate_accounts(
    rng: np.random.Generator, cfg: GeneratorConfig
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """One row per account with its 'home' device/ip/card, plus a few innocent
    household clusters that share a device — the honest FP-risk case for the graph layer.

    Returns the accounts table and the list of (account_a, account_b) household pairs,
    so the Day 3 ring detector can measure its false-positive rate against something
    real: innocent entity-sharing it should *not* flag as a coordinated ring.
    """
    account_ids = [f"acct_{i:06d}" for i in range(cfg.n_accounts)]
    devices = [_short_id("dev", rng) for _ in range(cfg.n_accounts)]
    ips = [_short_id("ip", rng) for _ in range(cfg.n_accounts)]
    cards = [_short_id("card", rng) for _ in range(cfg.n_accounts)]
    segments = rng.choice(SEGMENTS, size=cfg.n_accounts, p=SEGMENT_WEIGHTS)

    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "device_id": devices,
            "ip": ips,
            "card_fingerprint": cards,
            "segment": segments,
        }
    )

    # Innocent household sharing: pair up accounts and force a shared device_id, no fraud
    # implication. This is what lets the graph layer's FP rate be measured honestly later.
    n_household_accounts = int(cfg.n_accounts * cfg.household_sharing_rate)
    n_household_accounts -= n_household_accounts % 2
    household_pairs: list[tuple[str, str]] = []
    if n_household_accounts >= 2:
        idx = rng.choice(cfg.n_accounts, size=n_household_accounts, replace=False)
        for a, b in idx.reshape(-1, 2):
            accounts.loc[b, "device_id"] = accounts.loc[a, "device_id"]
            household_pairs.append((accounts.loc[a, "account_id"], accounts.loc[b, "account_id"]))

    return accounts, household_pairs


def generate_base_transactions(
    rng: np.random.Generator, accounts: pd.DataFrame, cfg: GeneratorConfig
) -> pd.DataFrame:
    """Ordinary, uncoordinated transactions with an independent per-transaction fraud
    label. Fraud rows are drawn from shifted distributions (heavier-tailed amount,
    odd-hour timestamps) so the point-risk model has real, learnable signal — not noise.
    """
    n = cfg.n_base_transactions
    acct_idx = rng.integers(0, len(accounts), size=n)
    txn_segments = accounts["segment"].to_numpy()[acct_idx]

    amount_mult = np.array([SEGMENT_PROFILES[s][0] for s in txn_segments])
    fraud_mult = np.array([SEGMENT_PROFILES[s][1] for s in txn_segments])
    is_fraud = rng.random(n) < np.clip(cfg.base_fraud_rate * fraud_mult, 0, 1)

    amounts = (
        np.where(
            is_fraud,
            rng.lognormal(mean=cfg.fraud_amount_mu, sigma=cfg.fraud_amount_sigma, size=n),
            rng.lognormal(mean=cfg.legit_amount_mu, sigma=cfg.legit_amount_sigma, size=n),
        )
        * amount_mult
    ).round(2)

    offsets_days = rng.random(n) * cfg.window_days
    # Legit traffic clusters in normal daytime hours; fraud skews toward off-hours.
    hour_legit = np.clip(rng.normal(14, 4, size=n), 0, 23)
    hour_fraud = np.mod(rng.normal(3, 2.5, size=n), 24)
    hours = np.where(is_fraud, hour_fraud, hour_legit)
    timestamps = [
        cfg.start_time + timedelta(days=float(d), hours=float(h))
        for d, h in zip(offsets_days, hours, strict=True)
    ]

    txns = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:08d}" for i in range(n)],
            "account_id": accounts.loc[acct_idx, "account_id"].to_numpy(),
            "device_id": accounts.loc[acct_idx, "device_id"].to_numpy(),
            "ip": accounts.loc[acct_idx, "ip"].to_numpy(),
            "card_fingerprint": accounts.loc[acct_idx, "card_fingerprint"].to_numpy(),
            "segment": txn_segments,
            "amount": amounts,
            "timestamp": timestamps,
            "label": is_fraud.astype(int),
            "ring_id": None,
        }
    )
    return txns


def inject_fraud_rings(
    rng: np.random.Generator, accounts: pd.DataFrame, cfg: GeneratorConfig
) -> tuple[pd.DataFrame, dict]:
    """Manufacture coordinated fraud rings: a handful of accounts forced to share a
    device/card, transacting in a short burst with amounts clustered just under a
    structuring threshold. Returns the injected transactions plus a ground-truth mapping
    ring_id -> account_ids, for validating the Day 3 graph layer.
    """
    ring_rows = []
    ground_truth: dict[str, list[str]] = {}
    used_accounts: set[int] = set()

    for r in range(cfg.n_rings):
        ring_id = f"ring_{r:03d}"
        size = rng.integers(cfg.ring_size_range[0], cfg.ring_size_range[1] + 1)
        available = np.setdiff1d(np.arange(len(accounts)), list(used_accounts))
        if len(available) < size:
            break
        member_idx = rng.choice(available, size=size, replace=False)
        used_accounts.update(member_idx.tolist())

        shared_device = _short_id("dev", rng)
        shared_card = _short_id("card", rng) if rng.random() < 0.5 else None

        burst_start_day = rng.uniform(0, cfg.window_days - 1)
        burst_minutes = rng.integers(*cfg.ring_burst_minutes_range)
        burst_start = cfg.start_time + timedelta(days=float(burst_start_day))

        member_account_ids = accounts.loc[member_idx, "account_id"].tolist()
        ground_truth[ring_id] = member_account_ids

        for acc_idx in member_idx:
            n_txns = rng.integers(*cfg.txns_per_ring_account_range)
            for _ in range(n_txns):
                offset_min = rng.uniform(0, burst_minutes)
                amount = cfg.structuring_threshold - abs(rng.normal(150, 60))
                ring_rows.append(
                    {
                        "transaction_id": f"txn_ring_{uuid.uuid4().hex[:10]}",
                        "account_id": accounts.loc[acc_idx, "account_id"],
                        "device_id": shared_device,
                        "ip": accounts.loc[acc_idx, "ip"],
                        "card_fingerprint": shared_card or accounts.loc[acc_idx, "card_fingerprint"],
                        "segment": accounts.loc[acc_idx, "segment"],
                        "amount": round(max(amount, 50.0), 2),
                        "timestamp": burst_start + timedelta(minutes=float(offset_min)),
                        "label": 1,
                        "ring_id": ring_id,
                    }
                )

    return pd.DataFrame(ring_rows), ground_truth


def build_entity_edges(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive an entity-link edge list (account <-> account) from shared device/ip/card
    across distinct accounts. This is what the Day 3 graph/Louvain layer consumes.
    """
    edges = []
    for entity_col, edge_type in (
        ("device_id", "shared_device"),
        ("card_fingerprint", "shared_card"),
        ("ip", "shared_ip"),
    ):
        grouped = transactions.groupby(entity_col)["account_id"].unique()
        for entity_value, accts in grouped.items():
            accts = sorted(set(accts))
            if len(accts) < 2:
                continue
            for i in range(len(accts)):
                for j in range(i + 1, len(accts)):
                    edges.append(
                        {
                            "entity_a": accts[i],
                            "entity_b": accts[j],
                            "edge_type": edge_type,
                            "shared_value": entity_value,
                        }
                    )
    if not edges:
        return pd.DataFrame(columns=["entity_a", "entity_b", "edge_type", "weight"])

    edges_df = pd.DataFrame(edges)
    weighted = (
        edges_df.groupby(["entity_a", "entity_b"])
        .size()
        .reset_index(name="weight")
    )
    return weighted


def generate_dataset(cfg: GeneratorConfig | None = None) -> dict:
    """End-to-end generation: accounts -> base transactions -> injected rings ->
    combined transaction table + entity edges + ring ground truth.
    """
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.random_seed)

    accounts, household_pairs = generate_accounts(rng, cfg)
    base_txns = generate_base_transactions(rng, accounts, cfg)
    ring_txns, rings_ground_truth = inject_fraud_rings(rng, accounts, cfg)

    all_txns = pd.concat([base_txns, ring_txns], ignore_index=True)
    all_txns = all_txns.sort_values("timestamp").reset_index(drop=True)

    entity_edges = build_entity_edges(all_txns)

    return {
        "accounts": accounts,
        "transactions": all_txns,
        "entity_edges": entity_edges,
        "rings_ground_truth": rings_ground_truth,
        "household_pairs": household_pairs,
    }
