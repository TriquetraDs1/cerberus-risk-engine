"""The real-data ring validation harness (scripts/validate_rings_real_data.py).

The dataset it consumes is a Kaggle competition download that cannot be fetched
automatically, so these tests exercise the graph-building logic against a small frame
shaped like IEEE-CIS instead. That matters: without them the harness would be untested
code waiting on a file, and the first real run would be debugging the harness rather than
measuring the detector.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_rings_real_data import (  # noqa: E402
    MAX_ACCOUNTS_PER_ENTITY,
    build_real_entity_edges,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_accounts_sharing_an_identifier_are_linked():
    edges = build_real_entity_edges(
        _frame(
            [
                {"account_id": "a", "addr1": 100, "card2": None, "DeviceInfo": None},
                {"account_id": "b", "addr1": 100, "card2": None, "DeviceInfo": None},
                {"account_id": "c", "addr1": 999, "card2": None, "DeviceInfo": None},
            ]
        )
    )
    pairs = {tuple(sorted((r.entity_a, r.entity_b))) for r in edges.itertuples()}
    assert ("a", "b") in pairs
    assert not any("c" in p for p in pairs)


def test_edge_weight_counts_distinct_shared_identifier_types():
    """Two accounts sharing both an address and a device are a stronger signal than two
    sharing only an address, and the weight has to carry that — Louvain reads it."""
    edges = build_real_entity_edges(
        _frame(
            [
                {"account_id": "a", "addr1": 1, "card2": None, "DeviceInfo": "iPhone"},
                {"account_id": "b", "addr1": 1, "card2": None, "DeviceInfo": "iPhone"},
                {"account_id": "c", "addr1": 2, "card2": None, "DeviceInfo": None},
                {"account_id": "d", "addr1": 2, "card2": None, "DeviceInfo": None},
            ]
        )
    )
    by_pair = {tuple(sorted((r.entity_a, r.entity_b))): r.weight for r in edges.itertuples()}
    assert by_pair[("a", "b")] == 2
    assert by_pair[("c", "d")] == 1


def test_category_like_identifiers_are_dropped():
    """The single most important difference between real and synthetic entity data.

    The generator's identifiers are unique by construction, so it never produces a device
    string shared by fifty thousand unrelated people. Real data does — "gmail.com", a card
    network, a common Android build — and linking on those collapses the graph into one
    component where every account is connected to every other. Without this cutoff the
    false-positive measurement would be meaningless rather than merely hard.
    """
    shared_by_many = [
        {"account_id": f"acct_{i}", "addr1": 7, "card2": None, "DeviceInfo": None}
        for i in range(MAX_ACCOUNTS_PER_ENTITY + 5)
    ]
    assert build_real_entity_edges(_frame(shared_by_many)).empty

    just_under = [
        {"account_id": f"acct_{i}", "addr1": 7, "card2": None, "DeviceInfo": None}
        for i in range(MAX_ACCOUNTS_PER_ENTITY)
    ]
    assert not build_real_entity_edges(_frame(just_under)).empty


def test_missing_identifier_values_do_not_link_accounts():
    """Null is not a shared entity. Grouping on it would link every account with an
    unknown device to every other, which is the same collapse as the category case."""
    edges = build_real_entity_edges(
        _frame(
            [
                {"account_id": "a", "addr1": None, "card2": None, "DeviceInfo": None},
                {"account_id": "b", "addr1": None, "card2": None, "DeviceInfo": None},
            ]
        )
    )
    assert edges.empty


def test_absent_columns_are_tolerated():
    """train_identity.csv is an optional separate download, so DeviceInfo may not exist.
    The harness must degrade to the columns it has rather than raising."""
    edges = build_real_entity_edges(
        _frame([{"account_id": "a", "addr1": 5}, {"account_id": "b", "addr1": 5}])
    )
    assert len(edges) == 1
