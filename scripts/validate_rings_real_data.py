#!/usr/bin/env python
"""Validate the ring detector's false-positive rate against real transaction data.

**The gap this closes.** Every ring-detection number in this repo has been measured
against synthetic households. `reports/ring_detection_report.json` reports that 22.3% of
innocent device-sharing gets wrongly escalated — but the "innocent households" it measures
against are ones this repo invented, so the number describes the generator as much as the
detector. `docs/ARCHITECTURE.md` section 7 names this as the standing unanswered critique.

**Why creditcard.csv cannot close it.** That dataset is `Time`, `V1`-`V28`, `Amount`,
`Class`: anonymised principal components with no card, device, or address field. There is
nothing to build an entity graph from, so it can validate a base rate and nothing about a
graph detector. It was never the right instrument for this question.

**The insight that makes this tractable.** Validating a *false-positive* rate does not
require fraud-ring ground truth — no public dataset has that, which is why the graph layer
has gone unvalidated. It requires real people who genuinely share identifiers and are not
a fraud ring. IEEE-CIS has hundreds of thousands of them: real cards, real billing
addresses, real device fingerprints, with a real `isFraud` label.

So the measurement is: build the entity graph the same way the synthetic pipeline does,
run the same detector, and ask **what fraction of flagged communities contain no fraud at
all.** Those are false positives on real entity-sharing — families on one card, offices
behind one billing address, people on a common device.

**What this still does not establish.** Recall. A community here carries no ring label, so
a flagged all-fraud community may be a genuine ring or a coincidence, and an unflagged one
may be a ring nobody labelled. This measures the false-positive side only — which is
precisely the side synthetic data was least able to speak to.

Usage:
    # Download from https://www.kaggle.com/competitions/ieee-fraud-detection/data
    # and put train_transaction.csv (+ optionally train_identity.csv) in data/raw/
    python scripts/validate_rings_real_data.py
    python scripts/validate_rings_real_data.py --max-rows 200000   # faster first pass
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cerberus.common.config import IEEE_IDENTITY_CSV, IEEE_TRANSACTION_CSV, REPORTS_DIR
from cerberus.detection.ring_detector import (
    COORDINATION_THRESHOLD,
    build_graph,
    detect_communities,
)

REAL_VALIDATION_JSON = REPORTS_DIR / "ring_validation_real_data.json"

# Identifiers treated as shared entities. Chosen to mirror what the synthetic generator
# links on (device / card / address). card1 is deliberately excluded: it is used as the
# account proxy below, so linking on it would only ever connect an account to itself.
LINK_FIELDS = ["addr1", "card2", "card3", "card5", "P_emaildomain", "DeviceInfo"]

# An identifier shared by more than this many accounts is not a shared *entity*, it is a
# category. "Visa", "gmail.com" or a common Android build links a large share of the
# dataset and would collapse the graph into one giant component.
#
# This is the single most important difference between real and synthetic entity data, and
# the reason a naive port of the pipeline produces nonsense: the generator's identifiers
# are unique by construction, so it never has to deal with a device string that fifty
# thousand unrelated people share. The cutoff is a judgement call and it materially changes
# the resulting graph, which is why it is recorded in the output.
MAX_ACCOUNTS_PER_ENTITY = 40


def build_real_entity_edges(txns: pd.DataFrame) -> pd.DataFrame:
    """Account-to-account edges from shared identifiers, weighted by how many distinct
    identifier types each pair shares — the same shape `build_entity_edges` produces for
    synthetic data, so the detector receives a graph of the kind it expects.
    """
    pair_weights: dict[tuple[str, str], int] = {}

    for field in LINK_FIELDS:
        if field not in txns.columns:
            continue
        grouped = txns.dropna(subset=[field]).groupby(field)["account_id"].unique()
        for accounts in grouped:
            if len(accounts) < 2 or len(accounts) > MAX_ACCOUNTS_PER_ENTITY:
                continue
            ordered = sorted(accounts)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    key = (ordered[i], ordered[j])
                    pair_weights[key] = pair_weights.get(key, 0) + 1

    if not pair_weights:
        return pd.DataFrame(columns=["entity_a", "entity_b", "weight"])
    return pd.DataFrame(
        [{"entity_a": a, "entity_b": b, "weight": w} for (a, b), w in pair_weights.items()]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Sample the first N transactions.")
    parser.add_argument("--coordination-threshold", type=float, default=COORDINATION_THRESHOLD)
    args = parser.parse_args()

    if not IEEE_TRANSACTION_CSV.exists():
        raise SystemExit(
            f"Missing {IEEE_TRANSACTION_CSV}.\n\n"
            "This validation needs a real dataset carrying entity identifiers.\n"
            "data/raw/creditcard.csv cannot be used: it is anonymised PCA components with\n"
            "no card, device, or address field, so no entity graph can be built from it.\n\n"
            "Download 'train_transaction.csv' (and optionally 'train_identity.csv') from\n"
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
            "and place them in data/raw/."
        )

    print(f"Loading {IEEE_TRANSACTION_CSV.name}...")
    wanted = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1", *LINK_FIELDS]
    header = pd.read_csv(IEEE_TRANSACTION_CSV, nrows=0).columns
    txns = pd.read_csv(
        IEEE_TRANSACTION_CSV,
        usecols=[c for c in wanted if c in header],
        nrows=args.max_rows,
    )

    if IEEE_IDENTITY_CSV.exists() and "DeviceInfo" not in txns.columns:
        ident = pd.read_csv(IEEE_IDENTITY_CSV, usecols=["TransactionID", "DeviceInfo"])
        txns = txns.merge(ident, on="TransactionID", how="left")

    # card1 is the closest thing IEEE-CIS has to a stable account identifier.
    txns["account_id"] = "ieee_" + txns["card1"].astype(str)
    txns["amount"] = txns["TransactionAmt"]
    # TransactionDT is seconds from an undocumented reference point. That is fine here:
    # the coordination score only ever reads relative spacing, never absolute dates.
    txns["timestamp"] = pd.to_datetime(txns["TransactionDT"], unit="s", origin="2017-12-01")

    n_accounts = txns["account_id"].nunique()
    print(
        f"  {len(txns):,} transactions, {n_accounts:,} distinct accounts, "
        f"{txns['isFraud'].mean():.3%} fraud rate"
    )

    print("Building the entity-link graph from shared card / address / device fields...")
    edges = build_real_entity_edges(txns)
    if edges.empty:
        raise SystemExit("No entity links found — nothing to validate.")
    graph = build_graph(edges)
    print(f"  {graph.number_of_nodes():,} linked accounts, {graph.number_of_edges():,} edges")

    print("Running the same detector the synthetic pipeline uses...")
    result = detect_communities(
        graph, transactions=txns, coordination_threshold=args.coordination_threshold
    )

    fraud_by_account = txns.groupby("account_id")["isFraud"].max()
    candidates = [cid for cid, m in result.communities.items() if len(m) >= 3]

    def has_fraud(community_id: int) -> bool:
        return any(fraud_by_account.get(m, 0) == 1 for m in result.communities[community_id])

    fraudy_flagged = sum(1 for cid in result.flagged_ring_ids if has_fraud(cid))
    clean_flagged = len(result.flagged_ring_ids) - fraudy_flagged
    n_flagged = len(result.flagged_ring_ids)
    fp_rate = clean_flagged / n_flagged if n_flagged else 0.0

    # How many entirely-clean communities existed to be wrongly flagged in the first place.
    clean_candidates = sum(1 for cid in candidates if not has_fraud(cid))
    clean_flag_rate = clean_flagged / clean_candidates if clean_candidates else 0.0

    print("\n--- Ring detection on real entity-sharing ---")
    print(f"Communities of size >= 3:                {len(candidates):,}")
    print(f"  of those, entirely non-fraud:          {clean_candidates:,}")
    print(f"Flagged as rings:                        {n_flagged:,}")
    print(f"  containing at least one fraud account: {fraudy_flagged:,}")
    print(f"  containing no fraud at all:            {clean_flagged:,}")
    print()
    print(f"False-positive rate among flagged:       {fp_rate:.1%}")
    print(f"Share of innocent communities flagged:   {clean_flag_rate:.1%}")
    print()
    print("The second number is the one comparable to the synthetic 22.3%: how often a")
    print("group of real people who share identifiers, with no fraud between them, gets")
    print("escalated as a ring.")

    REAL_VALIDATION_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "IEEE-CIS Fraud Detection (train_transaction.csv)",
                "n_transactions": int(len(txns)),
                "n_accounts": int(n_accounts),
                "observed_fraud_rate": float(txns["isFraud"].mean()),
                "link_fields": LINK_FIELDS,
                "max_accounts_per_entity": MAX_ACCOUNTS_PER_ENTITY,
                "coordination_threshold": args.coordination_threshold,
                "n_linked_accounts": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "n_candidate_communities": len(candidates),
                "n_clean_candidate_communities": clean_candidates,
                "n_flagged": n_flagged,
                "n_flagged_containing_fraud": fraudy_flagged,
                "n_flagged_entirely_clean": clean_flagged,
                "false_positive_rate_among_flagged": fp_rate,
                "innocent_community_flag_rate": clean_flag_rate,
                "synthetic_comparison": {
                    "household_false_positive_rate": 0.223,
                    "note": "From reports/ring_detection_report.json, measured on invented households.",
                },
                "limitations": [
                    "Measures the false-positive side only. IEEE-CIS carries no fraud-ring "
                    "labels — no public dataset does — so recall cannot be validated here, "
                    "and a flagged all-fraud community may be a real ring or a coincidence.",
                    "card1 is an account proxy, not an account: one person holding two "
                    "cards appears as two accounts, and a reissued card breaks continuity.",
                    "Identifiers shared by more than MAX_ACCOUNTS_PER_ENTITY accounts are "
                    "dropped as category-like. That cutoff is a judgement call and it "
                    "materially changes the graph.",
                    "TransactionDT has no documented origin, so absolute timestamps are "
                    "arbitrary; only relative spacing is used.",
                    "A community with no fraud label is assumed innocent. IEEE-CIS labels "
                    "confirmed fraud, so undetected fraud counts as clean here and this "
                    "false-positive rate is, if anything, an overestimate.",
                ],
            },
            indent=2,
        )
    )
    print(f"\nWrote {REAL_VALIDATION_JSON}")


if __name__ == "__main__":
    main()
