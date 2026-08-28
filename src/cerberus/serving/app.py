"""Day 7: the /score serving API.

Thin, enterprise-shaped — not enterprise-scale. Synchronous FastAPI, SQLite audit log,
in-process metrics and per-account history: exactly the demo-appropriate choices named
in docs/ARCHITECTURE.md's Trade-off Analysis, not a production claim. Run with:

    uvicorn cerberus.serving.app:app --reload

Then:
    curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{...}'
    curl localhost:8000/health
    curl localhost:8000/metrics
    curl localhost:8000/audit/recent
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import timedelta

import joblib
import lightgbm as lgb
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from cerberus.common.config import (
    BASELINE_MODEL_PATH,
    CALIBRATOR_PATH,
    DATA_PROCESSED,
    DETECTED_RINGS_JSON,
    MODELS_DIR,
    REPORTS_DIR,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.data.synthetic_rings import SEGMENTS
from cerberus.detection.explain import reason_codes_for_row
from cerberus.features.pipeline import FEATURE_COLUMNS, add_entity_degree
from cerberus.llm.narrate import DecisionContext, narrate_decision, narration_source
from cerberus.serving.audit import AuditLog
from cerberus.serving.logging_config import configure_logging, log_with_fields
from cerberus.serving.metrics import RING_CHECK_STATUS, SCORE_LATENCY, SCORE_REQUESTS, registry
from cerberus.serving.schemas import (
    CostBasis,
    ExplainResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from cerberus.serving.state import ServingState

DECISION_LAYER_JSON = REPORTS_DIR / "decision_layer.json"
AUDIT_DB_PATH = DATA_PROCESSED / "audit_log.db"

logger = configure_logging(logging.INFO)


class ModelBundle:
    """Everything the endpoint needs, loaded once at startup. Prefers the hardened
    model (Day 5-6) if it exists, falling back to the baseline — the service should
    serve the best artifact available, not require every day's script to have run.
    """

    def __init__(self) -> None:
        hardened_path = MODELS_DIR / "point_risk_hardened.txt"
        hardened_calibrator_path = MODELS_DIR / "point_risk_calibrator_hardened.joblib"

        if hardened_path.exists() and hardened_calibrator_path.exists():
            self.booster = lgb.Booster(model_file=str(hardened_path))
            self.calibrator = joblib.load(hardened_calibrator_path)
            self.model_version = "hardened"
        elif BASELINE_MODEL_PATH.exists():
            self.booster = lgb.Booster(model_file=str(BASELINE_MODEL_PATH))
            self.calibrator = joblib.load(CALIBRATOR_PATH)
            self.model_version = "baseline"
        else:
            raise RuntimeError(
                f"No trained model found at {BASELINE_MODEL_PATH} — run the pipeline "
                "scripts first (see README.md Quickstart)."
            )

        self.explainer = shap.TreeExplainer(self.booster)

        if not DECISION_LAYER_JSON.exists():
            raise RuntimeError(f"No decision layer found at {DECISION_LAYER_JSON} — run build_decision_layer.py first.")
        decision_layer = json.loads(DECISION_LAYER_JSON.read_text())
        self.segment_routing = decision_layer["segments"]
        self.global_default_threshold = decision_layer["global_default_threshold"]

        self.ring_membership: dict[str, str] = {}
        if DETECTED_RINGS_JSON.exists():
            detected = json.loads(DETECTED_RINGS_JSON.read_text())
            self.ring_membership = {a: rid for rid, members in detected.items() for a in members}

        self.entity_degree: dict[str, int] = {}
        if SYNTHETIC_ENTITY_EDGES_CSV.exists():
            edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
            degree_df = add_entity_degree(pd.DataFrame({"account_id": pd.concat([edges["entity_a"], edges["entity_b"]]).unique()}), edges)
            self.entity_degree = dict(zip(degree_df["account_id"], degree_df["entity_degree"], strict=True))

        self.global_amount_mean, self.global_amount_std = 0.0, 1.0
        if SYNTHETIC_TRANSACTIONS_CSV.exists():
            txns = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV, usecols=["amount"])
            self.global_amount_mean = float(txns["amount"].mean())
            self.global_amount_std = float(txns["amount"].std()) or 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelBundle()
    app.state.serving = ServingState()
    app.state.audit = AuditLog(AUDIT_DB_PATH)
    log_with_fields(logger, logging.INFO, "startup complete", model_version=app.state.model.model_version)
    yield


app = FastAPI(title="Cerberus Risk Engine", version="0.1.0", lifespan=lifespan)


def _build_feature_row(req: ScoreRequest, model: ModelBundle, state: ServingState) -> pd.DataFrame:
    history = state.get_history(req.account_id)
    history.record(req.timestamp, req.amount)  # inclusive-of-self, matching the offline pipeline's convention

    velocity_count, velocity_amount = history.trailing(req.timestamp, timedelta(hours=1))

    if len(history.amounts) >= 2:
        mean, std = history.mean_std()
    else:
        mean, std = model.global_amount_mean, model.global_amount_std
    amount_zscore = (req.amount - mean) / std

    segment = req.segment if req.segment in SEGMENTS else "digital_subscription"

    row = {
        "amount": req.amount,
        "amount_zscore": amount_zscore,
        "velocity_count_1h": float(velocity_count),
        "velocity_amount_1h": velocity_amount,
        "hour_of_day": req.timestamp.hour,
        "is_off_hours": int(0 <= req.timestamp.hour <= 5),
        "day_of_week": req.timestamp.weekday(),
        "entity_degree": model.entity_degree.get(req.account_id, 0),
    }
    for seg in SEGMENTS:
        row[f"segment_{seg}"] = int(segment == seg)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    model: ModelBundle = app.state.model
    state: ServingState = app.state.serving
    audit: AuditLog = app.state.audit

    X = _build_feature_row(req, model, state)
    raw_score = float(model.booster.predict(X)[0])
    calibrated_score = float(model.calibrator.predict([raw_score])[0])

    segment = req.segment if req.segment in SEGMENTS else "digital_subscription"
    routing = model.segment_routing.get(segment)
    block_threshold = routing["block_threshold"] if routing else model.global_default_threshold
    review_threshold = routing["review_threshold"] if routing else block_threshold * 0.5
    decision = "block" if calibrated_score >= block_threshold else "review" if calibrated_score >= review_threshold else "approve"

    # Graceful degradation: if the graph service is marked unavailable, skip the ring
    # lookup entirely rather than crashing or serving a stale/wrong ring_id — the
    # point-risk model still serves a decision on its own. See docs/ARCHITECTURE.md,
    # "Error handling / retry logic."
    if state.is_graph_available():
        ring_id = model.ring_membership.get(req.account_id)
        ring_check = "ok"
    else:
        ring_id = None
        ring_check = "unavailable"
    RING_CHECK_STATUS.labels(status=ring_check).inc()

    shap_row = model.explainer.shap_values(X)
    if isinstance(shap_row, list):
        shap_row = shap_row[1]
    reason_codes = reason_codes_for_row(shap_row[0], ring_id)
    if ring_check == "unavailable":
        reason_codes = [c for c in reason_codes if not c.startswith("shared_device_with_flagged_ring")]
        reason_codes.append("ring_check_unavailable")

    cost_basis = CostBasis(
        fp_cost=routing["cost_matrix"]["fp_cost"] if routing else 5.0,
        fn_cost=routing["cost_matrix"]["fn_cost"] if routing else 50.0,
        block_threshold=block_threshold,
        review_threshold=review_threshold,
    )

    response = ScoreResponse(
        transaction_id=req.transaction_id,
        risk_score=round(calibrated_score, 4),
        decision=decision,
        reason_codes=reason_codes,
        ring_id=ring_id,
        ring_check=ring_check,
        cost_basis=cost_basis,
        model_version=model.model_version,
    )

    audit.record(
        transaction_id=req.transaction_id,
        account_id=req.account_id,
        segment=segment,
        amount=req.amount,
        risk_score=response.risk_score,
        decision=decision,
        ring_id=ring_id,
        ring_check=ring_check,
        model_version=model.model_version,
        reason_codes=reason_codes,
    )

    SCORE_REQUESTS.labels(decision=decision).inc()
    elapsed = time.perf_counter() - start
    SCORE_LATENCY.observe(elapsed)
    log_with_fields(
        logger,
        logging.INFO,
        "scored transaction",
        transaction_id=req.transaction_id,
        decision=decision,
        risk_score=response.risk_score,
        ring_check=ring_check,
        latency_ms=round(elapsed * 1000, 2),
    )

    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model: ModelBundle = app.state.model
    state: ServingState = app.state.serving
    return HealthResponse(
        status="ok",
        model_version=model.model_version,
        graph_cache_status="fresh" if state.is_graph_available() else "degraded",
        n_segments_loaded=len(model.segment_routing),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(registry).decode("utf-8"))


@app.get("/audit/recent")
def audit_recent(limit: int = 50) -> list[dict]:
    audit: AuditLog = app.state.audit
    return audit.recent(limit=limit)


@app.get("/explain/{transaction_id}", response_model=ExplainResponse)
def explain(transaction_id: str) -> ExplainResponse:
    """A1: a plain-English summary of a decision already in the audit log. Rebuilt from
    the logged decision plus this segment's routing/cost basis — it describes the
    recorded outcome, it does not re-score. `reason_codes` are echoed so the caller can
    check the prose against the structured output.
    """
    audit: AuditLog = app.state.audit
    model: ModelBundle = app.state.model

    record = audit.get(transaction_id)
    if record is None:
        raise HTTPException(404, f"No scored transaction {transaction_id!r} in the audit log.")

    reason_codes = [c for c in (record["reason_codes"] or "").split(",") if c]
    routing = model.segment_routing.get(record["segment"])
    block_threshold = routing["block_threshold"] if routing else model.global_default_threshold
    review_threshold = routing["review_threshold"] if routing else block_threshold * 0.5

    ctx = DecisionContext(
        transaction_id=transaction_id,
        decision=record["decision"],
        risk_score=float(record["risk_score"]),
        reason_codes=tuple(reason_codes),
        ring_id=record["ring_id"],
        segment=record["segment"],
        amount=float(record["amount"]),
        fp_cost=routing["cost_matrix"]["fp_cost"] if routing else 5.0,
        fn_cost=routing["cost_matrix"]["fn_cost"] if routing else 50.0,
        block_threshold=block_threshold,
        review_threshold=review_threshold,
    )
    return ExplainResponse(
        transaction_id=transaction_id,
        explanation=narrate_decision(ctx),
        reason_codes=reason_codes,
        narration_source=narration_source(),
    )


@app.post("/admin/graph-status")
def set_graph_status(status: str) -> dict:
    """Demo endpoint for the graceful-degradation story: flips the in-memory graph
    availability flag so a reviewer can watch /score degrade and recover on command,
    rather than reading about it in a README.
    """
    if status not in ("fresh", "degraded"):
        raise HTTPException(422, "status must be 'fresh' or 'degraded'")
    state: ServingState = app.state.serving
    state.set_graph_status(status)
    log_with_fields(logger, logging.INFO, "graph status changed", status=status)
    return {"graph_cache_status": status}
