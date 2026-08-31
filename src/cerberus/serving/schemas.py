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
    # Roadmap B2. null when the optional sequence model isn't loaded, which is the
    # default deployment. It never lowers a decision — see cerberus.serving.ensemble.
    sequence_score: float | None = None
    sequence_escalated: bool = False
    # Roadmap B1, second opinion only — Louvain remains authoritative for ring_id.
    # null when the optional GNN isn't loaded. gnn_agrees_with_louvain is the field worth
    # watching: agreement is unremarkable, disagreement is what an analyst should look at.
    gnn_ring_score: float | None = None
    gnn_agrees_with_louvain: bool | None = None


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


class ExplainResponse(BaseModel):
    transaction_id: str
    explanation: str
    reason_codes: list[str]
    # "llm" if a live model wrote the text, "template" if the deterministic fallback did.
    # reason_codes are echoed so a caller can confirm the prose matches the structured
    # output it's meant to describe.
    narration_source: Literal["llm", "template"]


class DisputeRequest(BaseModel):
    """Optional body for /dispute.

    A dispute draft is a pure function of a decision context, so requiring the API to
    have scored the transaction itself is artificial coupling — the dashboard's queue is
    produced offline by the export script and those rows are legitimately absent from the
    serving audit log. Supplying the decision lets a caller draft for any decision it
    already holds.

    The response always states which of the two happened, because it matters: facts from
    the audit log were recorded by this service, and facts in a request body were not.
    """

    decision: Decision
    risk_score: float = Field(ge=0, le=1)
    segment: str
    amount: float = Field(gt=0)
    account_id: str
    reason_codes: list[str] = Field(default_factory=list)
    ring_id: str | None = None
    timestamp: str | None = None


class DisputeResponse(BaseModel):
    transaction_id: str
    draft: str
    reason_codes: list[str]
    source: Literal["llm", "template"]
    # "audit_log" = the service scored this itself and the facts are its own record.
    # "supplied"  = the caller provided the decision; the draft restates what it was told.
    facts_from: Literal["audit_log", "supplied"]


class CopilotMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class CopilotRequest(BaseModel):
    # The whole conversation each time: the API holds no session state, which keeps the
    # serving layer stateless and means a restart never strands a half-finished case chat.
    messages: list[CopilotMessage] = Field(min_length=1, max_length=24)


class CopilotResponse(BaseModel):
    ring_id: str
    answer: str
    n_members: int
    source: Literal["llm", "template"]
