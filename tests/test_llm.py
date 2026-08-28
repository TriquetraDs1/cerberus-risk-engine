"""A1 narration layer. No trained model needed — this is pure text assembly plus a
guarded optional LLM call, so it runs in CI unconditionally (CI has no API key, which
is exactly the templated path these tests pin down).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerberus.llm.narrate import DecisionContext, llm_enabled, narrate_batch, narrate_decision
from cerberus.llm.templates import humanize_reason_code, render_template

BLOCK_ROW = {
    "transaction_id": "txn_0001",
    "account_id": "acct_0001",
    "segment": "travel_luxury",
    "timestamp": "2026-02-19T07:09:14",
    "amount": 1899.0,
    "risk_score": 0.97,
    "decision": "block",
    "reason_codes": ["amount_anomalous_for_account", "shared_device_with_flagged_ring:detected_12"],
    "ring_id": "detected_12",
    "actual_label": 1,
    "cost_basis": {
        "fp_cost": 29.86,
        "fn_cost": 1517.79,
        "block_threshold": 0.0101,
        "review_threshold": 0.005,
    },
}

APPROVE_ROW = {
    **BLOCK_ROW,
    "transaction_id": "txn_0002",
    "decision": "approve",
    "risk_score": 0.002,
    "reason_codes": ["low_risk_no_dominant_factor"],
    "ring_id": None,
    "actual_label": 0,
}


def test_context_roundtrips_from_a_queue_row():
    ctx = DecisionContext.from_queue_row(BLOCK_ROW)
    assert ctx.transaction_id == "txn_0001"
    assert ctx.decision == "block"
    assert ctx.ring_id == "detected_12"
    assert ctx.fn_cost == pytest.approx(1517.79)
    assert ctx.reason_codes == tuple(BLOCK_ROW["reason_codes"])


def test_template_block_narration_is_grounded_and_clean():
    text = render_template(DecisionContext.from_queue_row(BLOCK_ROW))
    assert text.startswith("Blocked at a calibrated risk score of 0.97")
    assert "flagged ring detected_12" in text
    assert "1,518" in text  # fn_cost, humanized
    # internal-only codes never surface as prose
    assert "low_risk_no_dominant_factor" not in text
    assert "ring_check_unavailable" not in text
    assert "_" not in text.replace("detected_12", "")  # no leftover underscore_case


def test_template_approve_narration_cites_the_review_threshold():
    text = render_template(DecisionContext.from_queue_row(APPROVE_ROW))
    assert text.startswith("Approved")
    assert "0.005" in text  # review_threshold
    assert "cost matrix" not in text  # no block-rationale on an approve
    assert "No dominant risk factor" in text  # only reason code is the low-risk sentinel


def test_template_approve_with_minor_factors_still_frames_them_as_sub_threshold():
    row = {**APPROVE_ROW, "reason_codes": ["amount_anomalous_for_account", "unusual_hour"]}
    text = render_template(DecisionContext.from_queue_row(row))
    assert text.startswith("Approved")
    assert "strongest contributing factors" in text
    assert "below this segment's review threshold" in text
    assert "cost matrix" not in text


def test_humanize_keeps_the_ring_id_verbatim():
    assert humanize_reason_code("high_transaction_velocity") == "high transaction velocity"
    assert humanize_reason_code("shared_device_with_flagged_ring:detected_8") == (
        "shared device with flagged ring: detected_8"
    )


def test_narrate_decision_without_llm_equals_the_template():
    ctx = DecisionContext.from_queue_row(BLOCK_ROW)
    assert narrate_decision(ctx, use_llm=False) == render_template(ctx)


def test_llm_enabled_follows_the_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_enabled() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert llm_enabled() is True


def test_llm_path_falls_back_to_template_when_anthropic_is_unavailable(monkeypatch):
    """use_llm=True must never raise: no anthropic / a bad key degrades to the template,
    it does not break the export or the API response."""
    monkeypatch.setattr("cerberus.llm.narrate._llm_warning_emitted", False, raising=False)
    ctx = DecisionContext.from_queue_row(BLOCK_ROW)
    out = narrate_decision(ctx, use_llm=True)
    assert out == render_template(ctx)


def test_narrate_batch_reuses_one_result_per_transaction_id():
    rows = [BLOCK_ROW, BLOCK_ROW, APPROVE_ROW]
    out = narrate_batch([DecisionContext.from_queue_row(r) for r in rows], use_llm=False)
    assert len(out) == 3
    assert out[0] == out[1]
    assert out[2] != out[0]
