"""Smoke tests for the /score API. Skipped if the pipeline hasn't been run yet (no
trained model on disk) — this is a serving-layer test, not a training test; see
tests/test_synthetic_rings.py, test_ring_detector.py, and test_adversarial.py for
stage-specific coverage that doesn't need trained artifacts.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.common.config import BASELINE_MODEL_PATH, CALIBRATOR_PATH
from cerberus.serving.app import DECISION_LAYER_JSON

PIPELINE_READY = BASELINE_MODEL_PATH.exists() and CALIBRATOR_PATH.exists() and DECISION_LAYER_JSON.exists()

pytestmark = pytest.mark.skipif(
    not PIPELINE_READY,
    reason="Run generate_data.py, detect_rings.py, train_baseline.py, and "
    "build_decision_layer.py first — the serving layer needs a trained model.",
)


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    from cerberus.serving.app import app

    with TestClient(app) as c:
        yield c


def _sample_request(**overrides) -> dict:
    payload = {
        "transaction_id": "txn_test_001",
        "account_id": "acct_test_001",
        "device_id": "dev_test_001",
        "ip": "ip_test_001",
        "card_fingerprint": "card_test_001",
        "amount": 250.0,
        "timestamp": datetime(2026, 6, 1, 14, 30).isoformat(),
        "segment": "grocery_essentials",
    }
    payload.update(overrides)
    return payload


def test_health_reports_loaded_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] in ("baseline", "hardened")
    assert body["n_segments_loaded"] == 4


def test_score_returns_a_full_decision(client):
    resp = client.post("/score", json=_sample_request())
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("approve", "review", "block")
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["reason_codes"]
    assert body["cost_basis"]["fn_cost"] > 0
    assert body["ring_check"] == "ok"


def test_high_amount_travel_transaction_scores_higher_risk_than_small_grocery(client):
    """Not a claim about a specific number — a sanity check that the segment-aware
    decision layer is actually wired in: a huge, anomalous travel transaction should
    not score lower than a routine small grocery one.
    """
    small = client.post("/score", json=_sample_request(amount=45.0, segment="grocery_essentials")).json()
    large = client.post(
        "/score",
        json=_sample_request(
            transaction_id="txn_test_002",
            account_id="acct_test_002",
            amount=25000.0,
            segment="travel_luxury",
            timestamp=datetime(2026, 6, 1, 3, 15).isoformat(),
        ),
    ).json()
    assert large["risk_score"] >= small["risk_score"]


def test_graph_degradation_is_demoable(client):
    resp = client.post("/admin/graph-status", params={"status": "degraded"})
    assert resp.status_code == 200
    assert resp.json()["graph_cache_status"] == "degraded"

    scored = client.post("/score", json=_sample_request(transaction_id="txn_test_003")).json()
    assert scored["ring_check"] == "unavailable"
    assert scored["ring_id"] is None
    assert "ring_check_unavailable" in scored["reason_codes"]

    # restore, so this test doesn't leak state into others in the same process
    client.post("/admin/graph-status", params={"status": "fresh"})


def test_audit_log_records_every_score(client):
    client.post("/score", json=_sample_request(transaction_id="txn_test_audit"))
    resp = client.get("/audit/recent", params={"limit": 5})
    assert resp.status_code == 200
    records = resp.json()
    assert any(r["transaction_id"] == "txn_test_audit" for r in records)


def test_metrics_endpoint_is_prometheus_text(client):
    client.post("/score", json=_sample_request(transaction_id="txn_test_metrics"))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "cerberus_score_requests_total" in resp.text
