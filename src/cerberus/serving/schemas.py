"""Request/response schemas for the /score API — matches the contract in
docs/ARCHITECTURE.md section 2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["approve", "review", "block"]


class ScoreRequest(BaseModel):
    transaction_id: str
    account_id: str
    device_id: str
    ip: str
    card_fingerprint: str
    amount: float = Field(gt=0)
    timestamp: datetime
    segment: str = Field(
        default="digital_subscription",
        description="Merchant segment. Falls back to a default rather than rejecting "
        "the request if the caller doesn't have segmentation wired up yet.",
    )


class CostBasis(BaseModel):
    fp_cost: float
    fn_cost: float
    block_threshold: float
    review_threshold: float


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: Decision
    reason_codes: list[str]
    ring_id: str | None
    ring_check: Literal["ok", "unavailable"]
    cost_basis: CostBasis
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    graph_cache_status: Literal["fresh", "degraded"]
    n_segments_loaded: int


class AuditRecord(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    segment: str
    risk_score: float
    decision: Decision
    ring_id: str | None
    scored_at: str
