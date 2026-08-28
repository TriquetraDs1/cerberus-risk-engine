"""Deterministic fallback narration — the no-key path, and the path CI exercises.

Every sentence here is assembled purely from the same `DecisionContext` fields the LLM
prompt gets. If this drifts from what the LLM would say, this is the one that's correct:
it cannot hallucinate, because it only ever restates its inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cerberus.llm.narrate import DecisionContext

_DECISION_VERB = {
    "block": "Blocked",
    "review": "Flagged for review",
    "approve": "Approved",
}


def humanize_reason_code(code: str) -> str:
    """Mirror of the dashboard's formatReasonCode: the label before the first colon is
    underscore_case; anything after it (a ring id) is a real identifier, kept verbatim.
    """
    label, sep, rest = code.partition(":")
    pretty = label.replace("_", " ")
    return f"{pretty}: {rest}" if sep else pretty


def _segment_label(segment: str) -> str:
    return segment.replace("_", " ")


def render_template(ctx: DecisionContext) -> str:
    verb = _DECISION_VERB.get(ctx.decision, ctx.decision)
    parts = [f"{verb} at a calibrated risk score of {ctx.risk_score:.2f}."]

    reasons = [
        humanize_reason_code(c)
        for c in ctx.reason_codes
        if c not in ("low_risk_no_dominant_factor", "ring_check_unavailable")
    ]

    if ctx.decision == "approve":
        if reasons:
            parts.append(
                "The strongest contributing factors were "
                + ", ".join(reasons)
                + f", but the score stays below this segment's review threshold of "
                f"{ctx.review_threshold:.3f}."
            )
        else:
            parts.append(
                f"No dominant risk factor; the score sits well below this segment's "
                f"review threshold of {ctx.review_threshold:.3f}."
            )
        return " ".join(parts)

    if reasons:
        parts.append("Primary factors: " + ", ".join(reasons) + ".")
    if ctx.ring_id:
        parts.append(f"The account is linked to flagged ring {ctx.ring_id}.")
    parts.append(
        f"Under the {_segment_label(ctx.segment)} cost matrix (a missed fraud costs "
        f"about ₹{ctx.fn_cost:,.0f} versus about ₹{ctx.fp_cost:,.0f} for a wrong "
        f"block) the block threshold is set to {ctx.block_threshold:.3f}."
    )
    return " ".join(parts)
