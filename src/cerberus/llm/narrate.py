"""narrate_decision(ctx) -> str : an LLM sentence if a key is configured, a
deterministic templated sentence otherwise. Never raises; a failed LLM call falls
back to the template rather than breaking a batch export or an API response.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from cerberus.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from cerberus.llm.templates import render_template

# Cheap and fast — this is a short, low-stakes narration task, not analysis. Override
# with CERBERUS_NARRATION_MODEL if you want the phrasing from a larger model for a demo.
DEFAULT_MODEL = os.getenv("CERBERUS_NARRATION_MODEL", "claude-haiku-4-5")
_MAX_TOKENS = 220

_llm_warning_emitted = False


@dataclass(frozen=True)
class DecisionContext:
    """Everything the narration needs — all of it already produced by the pipeline.
    No SHAP, no model, no re-computation: this is built from a `/score` response or a
    row of `dashboard/public/data/queue.json`.
    """

    transaction_id: str
    decision: str
    risk_score: float
    reason_codes: tuple[str, ...]
    ring_id: str | None
    segment: str
    amount: float
    fp_cost: float
    fn_cost: float
    block_threshold: float
    review_threshold: float
    actual_label: int | None = None

    @classmethod
    def from_queue_row(cls, row: dict) -> DecisionContext:
        cb = row["cost_basis"]
        return cls(
            transaction_id=row["transaction_id"],
            decision=row["decision"],
            risk_score=float(row["risk_score"]),
            reason_codes=tuple(row.get("reason_codes") or ()),
            ring_id=row.get("ring_id"),
            segment=row["segment"],
            amount=float(row["amount"]),
            fp_cost=float(cb["fp_cost"]),
            fn_cost=float(cb["fn_cost"]),
            block_threshold=float(cb["block_threshold"]),
            review_threshold=float(cb["review_threshold"]),
            actual_label=row.get("actual_label"),
        )


def llm_enabled() -> bool:
    """True when a live narration call will be attempted. The one switch that decides
    whether `narrate_decision` returns model text or templated text.
    """
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def narration_source() -> str:
    return "llm" if llm_enabled() else "template"


def _warn_once(message: str) -> None:
    global _llm_warning_emitted
    if not _llm_warning_emitted:
        print(f"[cerberus.llm] {message} Falling back to templated narration.", file=sys.stderr)
        _llm_warning_emitted = True


def _call_claude(system: str, user: str, model: str, max_tokens: int = _MAX_TOKENS) -> str:
    """Shared Claude call for every generative surface (narration, dispute drafting,
    the case copilot). Never raises: on any failure the caller falls back to its own
    deterministic template, so a missing key or a network blip degrades the text rather
    than breaking the response."""
    try:
        import anthropic  # optional dependency, imported lazily
    except ModuleNotFoundError:
        _warn_once("`anthropic` is not installed (pip install 'cerberus[llm]').")
        return ""

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            _warn_once("LLM returned no text.")
        return text
    except Exception as exc:  # noqa: BLE001 - narration must never break its caller
        _warn_once(f"LLM call failed ({type(exc).__name__}: {exc}).")
        return ""


def narrate_decision(
    ctx: DecisionContext,
    *,
    model: str | None = None,
    use_llm: bool | None = None,
) -> str:
    """One 2-3 sentence explanation of `ctx`. Deterministic when `use_llm` is False or
    no `ANTHROPIC_API_KEY` is set; otherwise an LLM call with a template fallback.
    """
    if use_llm is None:
        use_llm = llm_enabled()
    if not use_llm:
        return render_template(ctx)

    text = _call_claude(SYSTEM_PROMPT, build_user_prompt(ctx), model or DEFAULT_MODEL)
    return text or render_template(ctx)


def narrate_batch(contexts: list[DecisionContext], **kwargs) -> list[str]:
    """Narrate many decisions, reusing one identical result per transaction id so a
    re-run of the export is deterministic and doesn't pay for duplicate calls.
    """
    cache: dict[str, str] = {}
    out = []
    for ctx in contexts:
        if ctx.transaction_id not in cache:
            cache[ctx.transaction_id] = narrate_decision(ctx, **kwargs)
        out.append(cache[ctx.transaction_id])
    return out
