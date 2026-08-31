"""A2: chargeback dispute-evidence drafting.

Closes the last unbuilt item in docs/ARCHITECTURE.md §1's functional requirements
("auto-draft chargeback dispute evidence for flagged transactions").

Same contract as the A1 narration layer, and for the same reasons: it reads a decision
the pipeline already made and writes prose about it. It never re-scores, never changes a
decision, and always has a deterministic templated fallback so the repo runs with no API
key and no network.

The difference from A1 is length and structure. A dispute submission is a formatted
argument a processor's reviewer reads, not a one-line summary, so the output is sectioned
and the evidence is enumerated. That also makes it more format-sensitive, which is why
`reports/sample_disputes/` holds hand-checked examples: a long generated document that
nobody has read end to end is a liability, not a feature.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from cerberus.llm.narrate import DecisionContext, _call_claude, llm_enabled
from cerberus.llm.templates import humanize_reason_code

DEFAULT_MODEL = os.getenv("CERBERUS_DISPUTE_MODEL", "claude-sonnet-5")
_MAX_TOKENS = 900

SYSTEM_PROMPT = (
    "You draft chargeback dispute evidence for a payments fraud team. You are given the "
    "structured output of a deterministic risk pipeline for one transaction, plus that "
    "account's recent transaction history. Write the evidence narrative the merchant "
    "would submit to the card network.\n\n"
    "Structure, in this order, using these exact headings:\n"
    "SUMMARY — two sentences: what was decided and the single strongest reason.\n"
    "EVIDENCE — a numbered list. One item per concrete, checkable fact from the input.\n"
    "RISK METHODOLOGY — two or three sentences on how the decision was reached, "
    "including the segment's cost basis and the threshold that applied.\n"
    "LIMITATIONS — one or two sentences naming what this evidence does not establish.\n\n"
    "Rules:\n"
    "- Use only facts present in the input. Never invent an amount, a date, a device "
    "identifier, a customer communication, or a delivery record.\n"
    "- Do not overstate. This is evidence, not advocacy; a reviewer who catches one "
    "inflated claim discounts the rest.\n"
    "- The LIMITATIONS section is mandatory and must be substantive.\n"
    "- Plain text. No markdown, no bullets other than the numbered evidence list."
)


@dataclass
class DisputeContext:
    """Everything a draft needs. Built from a scored decision plus account history —
    no new computation, no model call."""

    decision: DecisionContext
    account_id: str
    timestamp: str
    n_account_transactions: int = 0
    account_total_amount: float = 0.0
    recent_amounts: tuple[float, ...] = field(default_factory=tuple)
    ring_member_count: int | None = None


def _evidence_items(ctx: DisputeContext) -> list[str]:
    """The factual spine of the draft. The template renders these directly; the LLM is
    given the same list and asked to phrase them. Keeping one source for the facts is
    what stops the two paths from disagreeing about what happened."""
    d = ctx.decision
    items = [
        f"Transaction {d.transaction_id} on account {ctx.account_id} was scored at a "
        f"calibrated fraud probability of {d.risk_score:.4f} and routed to "
        f"{d.decision.upper()} at {ctx.timestamp}.",
        f"The applicable block threshold for the {d.segment.replace('_', ' ')} segment is "
        f"{d.block_threshold:.4f}; the review threshold is {d.review_threshold:.4f}.",
    ]
    for code in d.reason_codes:
        if code in ("low_risk_no_dominant_factor", "ring_check_unavailable"):
            continue
        items.append(f"Automated risk factor recorded at decision time: {humanize_reason_code(code)}.")
    if d.ring_id:
        members = (
            f" That community contains {ctx.ring_member_count} linked accounts."
            if ctx.ring_member_count
            else ""
        )
        items.append(
            f"The account shares a device, card or network identifier with other accounts "
            f"in flagged community {d.ring_id}.{members}"
        )
    if ctx.n_account_transactions > 1:
        items.append(
            f"The account has {ctx.n_account_transactions} transactions on record in this "
            f"window, totalling ₹{ctx.account_total_amount:,.2f}."
        )
    items.append(
        f"Transaction amount ₹{d.amount:,.2f}, against a segment cost basis of ₹{d.fn_cost:,.0f} "
        f"per missed fraud and ₹{d.fp_cost:,.0f} per incorrectly declined transaction."
    )
    return items


def render_template(ctx: DisputeContext) -> str:
    """Deterministic draft. The no-key path, and what CI exercises."""
    d = ctx.decision
    items = _evidence_items(ctx)
    numbered = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    strongest = next(
        (humanize_reason_code(c) for c in d.reason_codes if c != "low_risk_no_dominant_factor"),
        "an aggregate risk score above the segment threshold",
    )

    return f"""SUMMARY
Transaction {d.transaction_id} was routed to {d.decision.upper()} by automated risk scoring at a calibrated fraud probability of {d.risk_score:.4f}. The strongest single contributing factor was {strongest}.

EVIDENCE
{numbered}

RISK METHODOLOGY
The score is a calibrated probability, not a ranking: isotonic regression fitted on a held-out split maps model output to observed fraud rates. The routing threshold is cost-optimised per merchant segment rather than set globally, deriving from that segment's own false-positive and false-negative costs. This transaction exceeded the {d.segment.replace('_', ' ')} segment's threshold of {d.block_threshold:.4f}.

LIMITATIONS
This evidence establishes that the transaction matched automated risk patterns at the time of the decision. It does not establish cardholder intent, and it does not incorporate delivery confirmation, customer communications, or any manual review that may have followed. Fraud-ring linkage reflects shared technical identifiers, which legitimate users who share a device can also produce."""


def _build_user_prompt(ctx: DisputeContext) -> str:
    d = ctx.decision
    facts = "\n".join(f"- {item}" for item in _evidence_items(ctx))
    return (
        f"transaction_id: {d.transaction_id}\n"
        f"account_id: {ctx.account_id}\n"
        f"decision: {d.decision}\n"
        f"calibrated_risk_score: {d.risk_score:.4f}\n"
        f"merchant_segment: {d.segment}\n"
        f"amount: ₹{d.amount:,.2f}\n"
        f"decided_at: {ctx.timestamp}\n"
        f"ring_linkage: {d.ring_id or 'none detected'}\n"
        f"segment_cost_basis: ₹{d.fn_cost:,.0f} per missed fraud, ₹{d.fp_cost:,.0f} per wrong block\n"
        f"thresholds: block >= {d.block_threshold:.4f}, review >= {d.review_threshold:.4f}\n\n"
        f"Established facts — use these and nothing else:\n{facts}"
    )


def draft_dispute(ctx: DisputeContext, *, model: str | None = None, use_llm: bool | None = None) -> str:
    """A dispute evidence draft. Deterministic when no key is configured; a failed LLM
    call falls back to the template rather than breaking the caller."""
    if use_llm is None:
        use_llm = llm_enabled()
    if not use_llm:
        return render_template(ctx)

    text = _call_claude(SYSTEM_PROMPT, _build_user_prompt(ctx), model or DEFAULT_MODEL, max_tokens=_MAX_TOKENS)
    if not text:
        print("[cerberus.llm] dispute draft fell back to the template.", file=sys.stderr)
    return text or render_template(ctx)
