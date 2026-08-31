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
import math
import os
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
    RING_DETECTION_REPORT_JSON,
    SYNTHETIC_ENTITY_EDGES_CSV,
    SYNTHETIC_RINGS_JSON,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from cerberus.data.synthetic_rings import SEGMENTS
from cerberus.detection.explain import reason_codes_for_row
from cerberus.features.pipeline import (
    FEATURE_COLUMNS,
    VELOCITY_WINDOWS,
    add_entity_degree,
    add_graph_features,
)
from cerberus.llm.copilot import RingCase, answer_case_question
from cerberus.llm.dispute import DisputeContext, draft_dispute
from cerberus.llm.narrate import DecisionContext, narrate_decision, narration_source
from cerberus.serving.audit import AuditLog
from cerberus.serving.ensemble import apply_sequence_opinion
from cerberus.serving.logging_config import configure_logging, log_with_fields
from cerberus.serving.metrics import (
    RING_CHECK_STATUS,
    SCORE_LATENCY,
    SCORE_REQUESTS,
    SEQUENCE_ESCALATIONS,
    registry,
)
from cerberus.serving.schemas import (
    CopilotRequest,
    CopilotResponse,
    CostBasis,
    DisputeResponse,
    ExplainResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from cerberus.serving.state import ServingState

DECISION_LAYER_JSON = REPORTS_DIR / "decision_layer.json"
AUDIT_DB_PATH = DATA_PROCESSED / "audit_log.db"

# The offline pipeline names its windows as pandas offset strings; the serving path
# needs real timedeltas for the same spans. One mapping so the two can't drift.
_WINDOW_TIMEDELTAS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

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

        # Ground-truth ring membership, so the copilot can say whether a detected
        # community corresponds to an injected ring. Without it the only honest answer is
        # "unknown" — and asserting "no match" when ground truth was never loaded would be
        # a claim the service cannot support. Synthetic-data only; a production deployment
        # has no such file and the copilot correctly reports unknown.
        self.ground_truth_membership: dict[str, str] = {}
        if SYNTHETIC_RINGS_JSON.exists():
            truth = json.loads(SYNTHETIC_RINGS_JSON.read_text())
            self.ground_truth_membership = {a: rid for rid, members in truth.items() for a in members}

        self.entity_degree: dict[str, int] = {}
        self.entity_strength: dict[str, float] = {}
        self.component_size: dict[str, int] = {}
        if SYNTHETIC_ENTITY_EDGES_CSV.exists():
            edges = pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV)
            accounts = pd.DataFrame(
                {"account_id": pd.concat([edges["entity_a"], edges["entity_b"]]).unique()}
            )
            degree_df = add_entity_degree(accounts, edges)
            self.entity_degree = dict(zip(degree_df["account_id"], degree_df["entity_degree"], strict=True))
            # Same graph features the offline pipeline computes, precomputed once at
            # startup from the batch-built entity graph — the cached-lookup design named
            # in docs/ARCHITECTURE.md §3 ("Entity graph caching").
            graph_df = add_graph_features(accounts, edges)
            self.entity_strength = dict(
                zip(graph_df["account_id"], graph_df["shared_entity_strength"], strict=True)
            )
            self.component_size = dict(
                zip(graph_df["account_id"], graph_df["component_size"], strict=True)
            )

        # Sequence model (roadmap B2). Entirely optional: it needs torch and a trained
        # checkpoint, and the service must run without either. When absent, /score
        # behaves exactly as before and reports sequence_score: null.
        self.sequence_model = None
        self.sequence_calibrator = None
        seq_path = MODELS_DIR / "sequence_risk.pt"
        seq_calib_path = MODELS_DIR / "sequence_risk_calibrator.joblib"
        if seq_path.exists() and seq_calib_path.exists():
            try:
                import torch

                from cerberus.detection.sequence_risk import build_model

                model = build_model()
                model.load_state_dict(torch.load(seq_path, map_location="cpu"))
                model.eval()
                self.sequence_model = model
                self.sequence_calibrator = joblib.load(seq_calib_path)
            except Exception as exc:  # noqa: BLE001 - an optional signal must never block startup
                log_with_fields(
                    logger, logging.WARNING, "sequence model unavailable", error=f"{type(exc).__name__}: {exc}"
                )

        # GNN ring detector (roadmap B1), as a SECOND OPINION only — never authoritative.
        # docs/EXPERIMENT_ADVANCED_TRAINING.md records that its perfect held-out score is
        # matched exactly by a `degree >= 2` threshold, so treating it as the ring
        # detector would be promoting a number that measures the dataset's easiness.
        # Louvain stays the decision-maker; this exists so an analyst can see when the two
        # disagree, which is the genuinely informative signal.
        #
        # Transductive, so scores for every account are computed once here against the
        # whole cached graph rather than per request.
        self.gnn_ring_score: dict[str, float] = {}
        gnn_path = MODELS_DIR / "gnn_ring.pt"
        if gnn_path.exists() and SYNTHETIC_ENTITY_EDGES_CSV.exists():
            try:
                import torch

                from cerberus.detection.gnn_ring import (
                    build_edge_index,
                    build_model,
                    build_node_features,
                    predict_ring_scores,
                )
                from cerberus.detection.ring_detector import build_graph

                graph = build_graph(pd.read_csv(SYNTHETIC_ENTITY_EDGES_CSV))
                features, node_index = build_node_features(graph)
                edge_index = build_edge_index(graph, node_index)
                gnn = build_model(in_channels=features.shape[1])
                gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
                gnn.eval()
                scores = predict_ring_scores(gnn, features, edge_index)
                self.gnn_ring_score = {a: float(scores[i]) for a, i in node_index.items()}
            except Exception as exc:  # noqa: BLE001 - optional signal, never blocks startup
                log_with_fields(
                    logger, logging.WARNING, "gnn ring detector unavailable", error=f"{type(exc).__name__}: {exc}"
                )

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
    # Read once at startup so the copilot can ground an answer about false positives in
    # the detector's measured rate rather than an adjective.
    app.state.household_fp_rate = None
    if RING_DETECTION_REPORT_JSON.exists():
        app.state.household_fp_rate = json.loads(RING_DETECTION_REPORT_JSON.read_text()).get(
            "household_false_positive_rate"
        )
    log_with_fields(logger, logging.INFO, "startup complete", model_version=app.state.model.model_version)
    yield


app = FastAPI(title="Cerberus Risk Engine", version="0.1.0", lifespan=lifespan)

# Off by default (no behaviour change for the local demo). Set CERBERUS_CORS_ORIGINS to a
# comma-separated origin list — e.g. the deployed dashboard's URL — to let a browser
# frontend call this API directly. See DEPLOYMENT.md.
_cors_origins = [o.strip() for o in os.getenv("CERBERUS_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def _build_feature_row(req: ScoreRequest, model: ModelBundle, state: ServingState) -> pd.DataFrame:
    history = state.get_history(req.account_id)
    history.record(req.timestamp, req.amount)  # inclusive-of-self, matching the offline pipeline's convention

    if len(history.amounts) >= 2:
        mean, std = history.mean_std()
    else:
        mean, std = model.global_amount_mean, model.global_amount_std
    amount_zscore = (req.amount - mean) / std

    # Mirror of the offline add_trailing_amount_features fallbacks: a first-ever
    # transaction gets a neutral ratio of 1.0 and a long-quiet-period gap, not 0/NaN.
    trailing_mean = history.trailing_mean_excluding_current()
    amount_vs_trailing_mean = req.amount / (trailing_mean or req.amount or 1e-9)
    hours_since_last = history.hours_since_previous()
    if hours_since_last is None:
        hours_since_last = 24.0 * 30

    segment = req.segment if req.segment in SEGMENTS else "digital_subscription"
    radians = 2 * math.pi * req.timestamp.hour / 24.0

    row = {
        "amount": req.amount,
        "amount_zscore": amount_zscore,
        "amount_vs_trailing_mean": amount_vs_trailing_mean,
        "hours_since_last_txn": hours_since_last,
        "hour_of_day": req.timestamp.hour,
        "hour_sin": math.sin(radians),
        "hour_cos": math.cos(radians),
        "is_off_hours": int(0 <= req.timestamp.hour <= 5),
        "day_of_week": req.timestamp.weekday(),
        "entity_degree": model.entity_degree.get(req.account_id, 0),
        "shared_entity_strength": model.entity_strength.get(req.account_id, 0.0),
        "component_size": model.component_size.get(req.account_id, 1),
    }
    # Every trailing window the offline pipeline computes, from the same in-process
    # history — so the live model sees the same shape of input it was trained on.
    for window in VELOCITY_WINDOWS:
        count, total = history.trailing(req.timestamp, _WINDOW_TIMEDELTAS[window])
        row[f"velocity_count_{window}"] = float(count)
        row[f"velocity_amount_{window}"] = total

    for seg in SEGMENTS:
        row[f"segment_{seg}"] = int(segment == seg)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def _sequence_score(req: ScoreRequest, model: ModelBundle, state: ServingState) -> float | None:
    """Calibrated sequence-model score for this account's recent history, or None when
    the model isn't loaded. Builds the same window shape the offline trainer used, from
    the live per-account history rather than a dataframe."""
    if model.sequence_model is None:
        return None
    try:
        import numpy as np
        import torch

        from cerberus.features.sequences import N_SEQUENCE_FEATURES, SEQUENCE_LENGTH

        history = state.get_history(req.account_id)
        window = np.zeros((1, SEQUENCE_LENGTH, N_SEQUENCE_FEATURES), dtype=np.float32)

        # Most recent SEQUENCE_LENGTH transactions, oldest first, left-padded — matching
        # features/sequences.build_sequences exactly. A mismatch here would feed the model
        # a shape it never trained on and produce a confident, meaningless number.
        amounts = history.amounts[-SEQUENCE_LENGTH:]
        stamps = history.timestamps[-SEQUENCE_LENGTH:]
        offset = SEQUENCE_LENGTH - len(amounts)
        for i, (ts, amount) in enumerate(zip(stamps, amounts, strict=True)):
            prev = stamps[i - 1] if i > 0 else None
            gap_hours = (ts - prev).total_seconds() / 3600.0 if prev else 24.0 * 30
            radians = 2 * math.pi * ts.hour / 24.0
            row = [
                math.log1p(amount),
                math.log1p(max(gap_hours, 0.0)),
                math.sin(radians),
                math.cos(radians),
                *(1.0 if req.segment == seg else 0.0 for seg in SEGMENTS),
            ]
            window[0, offset + i, :] = row

        with torch.no_grad():
            raw = float(torch.sigmoid(model.sequence_model(torch.from_numpy(window)))[0])
        return float(model.sequence_calibrator.predict([raw])[0])
    except Exception as exc:  # noqa: BLE001 - a second opinion must never break scoring
        log_with_fields(logger, logging.WARNING, "sequence scoring failed", error=f"{type(exc).__name__}: {exc}")
        return None


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
    gnn_score = None
    gnn_agrees = None
    if state.is_graph_available():
        ring_id = model.ring_membership.get(req.account_id)
        ring_check = "ok"
        if model.gnn_ring_score:
            gnn_score = model.gnn_ring_score.get(req.account_id, 0.0)
            # Agreement between two independent detectors, not a verdict. Disagreement is
            # the interesting case and is what an analyst should see.
            gnn_agrees = (gnn_score >= 0.5) == (ring_id is not None)
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

    # Roadmap B2, escalate-only. See cerberus.serving.ensemble for why this raises
    # caution rather than blending into the score: the thresholds above were fitted on
    # the point-risk distribution, so routing a blended score through them would be
    # applying a boundary to a distribution it was never calibrated for.
    outcome = apply_sequence_opinion(decision, reason_codes, _sequence_score(req, model, state))
    decision, reason_codes = outcome.decision, outcome.reason_codes
    if outcome.escalated:
        SEQUENCE_ESCALATIONS.inc()

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
        sequence_score=round(outcome.sequence_score, 4) if outcome.sequence_score is not None else None,
        sequence_escalated=outcome.escalated,
        gnn_ring_score=round(gnn_score, 4) if gnn_score is not None else None,
        gnn_agrees_with_louvain=gnn_agrees,
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

    ctx, reason_codes = _decision_context_from_audit(record, model)
    return ExplainResponse(
        transaction_id=transaction_id,
        explanation=narrate_decision(ctx),
        reason_codes=reason_codes,
        narration_source=narration_source(),
    )


def _decision_context_from_audit(record: dict, model: ModelBundle) -> tuple[DecisionContext, list[str]]:
    """Rebuild the scoring context from a logged decision. Shared by /explain, /dispute
    and anything else that describes a decision after the fact, so all of them describe
    the same recorded outcome rather than each reconstructing it slightly differently."""
    reason_codes = [c for c in (record["reason_codes"] or "").split(",") if c]
    routing = model.segment_routing.get(record["segment"])
    block_threshold = routing["block_threshold"] if routing else model.global_default_threshold
    review_threshold = routing["review_threshold"] if routing else block_threshold * 0.5
    ctx = DecisionContext(
        transaction_id=record["transaction_id"],
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
    return ctx, reason_codes


@app.post("/dispute/{transaction_id}", response_model=DisputeResponse)
def dispute(transaction_id: str) -> DisputeResponse:
    """A2: draft chargeback dispute evidence for a decision already in the audit log.

    POST rather than GET because drafting is a billable generative call, and GET should
    stay safe to retry, prefetch and cache. Nothing is written server-side.
    """
    audit: AuditLog = app.state.audit
    model: ModelBundle = app.state.model

    record = audit.get(transaction_id)
    if record is None:
        raise HTTPException(404, f"No scored transaction {transaction_id!r} in the audit log.")

    ctx, reason_codes = _decision_context_from_audit(record, model)
    history = app.state.serving.get_history(record["account_id"])
    ring_members = [a for a, rid in model.ring_membership.items() if rid == record["ring_id"]]

    draft = draft_dispute(
        DisputeContext(
            decision=ctx,
            account_id=record["account_id"],
            timestamp=record["scored_at"],
            n_account_transactions=len(history.amounts),
            account_total_amount=float(sum(history.amounts)),
            recent_amounts=tuple(history.amounts[-5:]),
            ring_member_count=len(ring_members) or None,
        )
    )
    return DisputeResponse(
        transaction_id=transaction_id,
        draft=draft,
        reason_codes=reason_codes,
        source=narration_source(),
    )


@app.post("/copilot/{ring_id}", response_model=CopilotResponse)
def copilot(ring_id: str, req: CopilotRequest) -> CopilotResponse:
    """A3: answer an analyst's question about one detected ring.

    Read-only by construction — the copilot has no tools and cannot change any state, so
    a prompt injection in the case data has nothing to escalate to. See
    cerberus.llm.copilot for the full reasoning.
    """
    model: ModelBundle = app.state.model
    audit: AuditLog = app.state.audit

    members = sorted(a for a, rid in model.ring_membership.items() if rid == ring_id)
    if not members:
        raise HTTPException(404, f"No detected ring {ring_id!r}.")

    member_set = set(members)
    case_transactions = [
        {
            "transaction_id": r["transaction_id"],
            "account_id": r["account_id"],
            "amount": r["amount"],
            "risk_score": r["risk_score"],
            "decision": r["decision"],
            "segment": r["segment"],
            "scored_at": r["scored_at"],
        }
        for r in audit.recent(limit=500)
        if r["account_id"] in member_set
    ]

    answer = answer_case_question(
        RingCase(
            ring_id=ring_id,
            member_account_ids=members,
            n_edges=sum(model.entity_degree.get(a, 0) for a in members) // 2,
            ground_truth_ring_id=next(
                (model.ground_truth_membership[a] for a in members if a in model.ground_truth_membership),
                None,
            ),
            ground_truth_available=bool(model.ground_truth_membership),
            transactions=case_transactions,
            household_false_positive_rate=app.state.household_fp_rate,
        ),
        [m.model_dump() for m in req.messages],
    )
    return CopilotResponse(
        ring_id=ring_id,
        answer=answer,
        n_members=len(members),
        source=narration_source(),
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
