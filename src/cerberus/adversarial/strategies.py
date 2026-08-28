"""Evasion strategies the adversarial harness searches over.

Defensive-only, per the project-wide scope statement in README.md: every function
here generates *synthetic* transactions for a *sandboxed* ring of freshly-minted
synthetic accounts, and hands them to this repo's own detectors to measure whether
they still catch the ring. Nothing here touches a real system, a real account, or a
real payment instrument, and nothing here is reusable as real-world evasion tooling —
it's parametrized synthetic-data generation plus a search loop, not an exploit.

Three strategies, matching real fraud-ring behavior a static detector would miss:
  1. Structuring   — split large transactions into many smaller ones, each below the
                      block/review threshold, betting the model only looks per-transaction.
  2. Identity rotation — spread the ring across more distinct devices/cards instead of
                      sharing one, betting the graph layer needs a dense shared entity
                      to form a community.
  3. Slow ramp      — stretch the coordinated burst over a much longer time window,
                      betting the velocity features only catch fast bursts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cerberus.data.synthetic_rings import SEGMENT_PROFILES, SEGMENTS, GeneratorConfig


def _short_id(prefix: str, rng: np.random.Generator) -> str:
    raw = rng.integers(0, 2**32 - 1, size=4, dtype=np.uint32).tobytes()
    return f"{prefix}_{uuid.UUID(bytes=raw).hex[:10]}"


@dataclass
class EvasionRing:
    """A single sandboxed synthetic ring: its own fresh accounts and transactions,
    self-contained so the harness can score it without touching the main dataset.
    """

    transactions: pd.DataFrame
    member_account_ids: list[str]
    strategy: str
    params: dict = field(default_factory=dict)


def _fresh_accounts(rng: np.random.Generator, ring_size: int, segment: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": [f"advadv_{_short_id('acct', rng)}" for _ in range(ring_size)],
            "device_id": [_short_id("dev", rng) for _ in range(ring_size)],
            "ip": [_short_id("ip", rng) for _ in range(ring_size)],
            "card_fingerprint": [_short_id("card", rng) for _ in range(ring_size)],
            "segment": segment,
        }
    )


def generate_baseline_ring(
    rng: np.random.Generator,
    ring_size: int = 6,
    segment: str | None = None,
    cfg: GeneratorConfig | None = None,
) -> EvasionRing:
    """The naive ring shape from Day 3's generator: one shared device, a tight burst,
    amounts structured just under the reporting threshold. This is the attacker's
    starting point — already known to be caught ~100% of the time (see Day 3 report) —
    and every evasion strategy below is a transformation *away* from this baseline.
    """
    cfg = cfg or GeneratorConfig()
    segment = segment or str(rng.choice(SEGMENTS))
    accounts = _fresh_accounts(rng, ring_size, segment)
    shared_device = accounts.iloc[0]["device_id"]

    amount_mult = SEGMENT_PROFILES[segment][0]
    burst_start = datetime(2026, 6, 1) + timedelta(hours=float(rng.uniform(0, 24 * 30)))
    burst_minutes = rng.integers(*cfg.ring_burst_minutes_range)

    rows = []
    for _, acct in accounts.iterrows():
        n_txns = rng.integers(*cfg.txns_per_ring_account_range)
        for _ in range(n_txns):
            offset_min = rng.uniform(0, burst_minutes)
            amount = (cfg.structuring_threshold - abs(rng.normal(150, 60))) * amount_mult / 1.0
            rows.append(
                {
                    "transaction_id": f"txn_adv_{uuid.uuid4().hex[:10]}",
                    "account_id": acct["account_id"],
                    "device_id": shared_device,  # naive: everyone shares one device
                    "ip": acct["ip"],
                    "card_fingerprint": acct["card_fingerprint"],
                    "segment": segment,
                    "amount": round(max(amount, 50.0), 2),
                    "timestamp": burst_start + timedelta(minutes=float(offset_min)),
                    "label": 1,
                }
            )

    return EvasionRing(
        transactions=pd.DataFrame(rows),
        member_account_ids=accounts["account_id"].tolist(),
        strategy="baseline",
        params={},
    )


def apply_structuring(ring: EvasionRing, rng: np.random.Generator, n_splits: float, jitter: float) -> EvasionRing:
    """Split each transaction's amount into `n_splits` smaller transactions, spread a
    few minutes apart, each perturbed by `jitter` — betting the point-risk model's
    per-transaction amount/velocity features stay under threshold if no single
    transaction looks large or if the split transactions land in different 1h windows.
    """
    n_splits = max(1, round(n_splits))
    rows = []
    for _, txn in ring.transactions.iterrows():
        piece = txn["amount"] / n_splits
        for k in range(n_splits):
            noisy_amount = max(piece * (1 + rng.uniform(-jitter, jitter)), 10.0)
            rows.append(
                {
                    **txn.to_dict(),
                    "transaction_id": f"txn_adv_{uuid.uuid4().hex[:10]}",
                    "amount": round(noisy_amount, 2),
                    "timestamp": txn["timestamp"] + timedelta(minutes=float(k) * rng.uniform(5, 25)),
                }
            )
    return EvasionRing(
        transactions=pd.DataFrame(rows),
        member_account_ids=ring.member_account_ids,
        strategy="structuring",
        params={"n_splits": n_splits, "jitter": jitter},
    )


def apply_identity_rotation(ring: EvasionRing, rng: np.random.Generator, n_devices: float) -> EvasionRing:
    """Spread the ring across `n_devices` distinct devices instead of one shared
    device — betting the entity-link graph never forms a dense-enough community to
    cross Louvain's min_ring_size if no single device (or card) links enough members.
    """
    n_devices = max(1, round(n_devices))
    member_ids = ring.member_account_ids
    device_pool = [_short_id("dev", rng) for _ in range(min(n_devices, len(member_ids)))]
    device_for_account = {acct: device_pool[i % len(device_pool)] for i, acct in enumerate(member_ids)}

    txns = ring.transactions.copy()
    txns["device_id"] = txns["account_id"].map(device_for_account)
    return EvasionRing(
        transactions=txns,
        member_account_ids=member_ids,
        strategy="identity_rotation",
        params={"n_devices": n_devices},
    )


def apply_slow_ramp(ring: EvasionRing, rng: np.random.Generator, ramp_multiplier: float) -> EvasionRing:
    """Stretch the whole burst over `ramp_multiplier`x the original duration — betting
    the velocity features (transaction count/amount in a trailing 1h window) never
    trigger if the ring's transactions are spread far enough apart in time.
    """
    txns = ring.transactions.copy()
    if len(txns) == 0:
        return EvasionRing(txns, ring.member_account_ids, "slow_ramp", {"ramp_multiplier": ramp_multiplier})

    t0 = txns["timestamp"].min()
    txns["timestamp"] = t0 + (txns["timestamp"] - t0) * ramp_multiplier
    # re-jitter within the stretched window so transactions don't land on the exact
    # same stretched instants they started at
    jitter_minutes = rng.uniform(-5, 5, size=len(txns))
    txns["timestamp"] = txns["timestamp"] + pd.to_timedelta(jitter_minutes, unit="m")
    return EvasionRing(
        transactions=txns,
        member_account_ids=ring.member_account_ids,
        strategy="slow_ramp",
        params={"ramp_multiplier": ramp_multiplier},
    )


# Search space per strategy: (low, high) bounds the adaptive search explores.
STRATEGY_PARAM_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "structuring": {"n_splits": (1, 10), "jitter": (0.0, 0.4)},
    "identity_rotation": {"n_devices": (1, 8)},
    "slow_ramp": {"ramp_multiplier": (1.0, 40.0)},
}

STRATEGY_APPLIERS = {
    "structuring": apply_structuring,
    "identity_rotation": apply_identity_rotation,
    "slow_ramp": apply_slow_ramp,
}
