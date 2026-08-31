"""Live integration of the sequence model (roadmap B2) into the serving path.

**Escalate-only.** The sequence model can move a decision toward caution
(approve -> review, review -> block) and can never move it the other way.

That constraint is not timidity, it is the only correct option given how the thresholds
were built. `reports/decision_layer.json` holds per-segment thresholds fitted on the
*calibrated point-risk* score distribution. Routing on a blended score would apply those
thresholds to a distribution they were never fitted for — the numbers would still print,
and they would be wrong. Refitting on the blend is the proper fix and is a real piece of
work (`scripts/build_decision_layer.py` would need a matching ensemble mode, and every
documented figure moves with it).

Until then, escalate-only is sound in a way blending is not: raising caution on a second
opinion needs no threshold calibration, because it never claims the blended score crossed
a fitted boundary. It only claims a second model disagreed in the risky direction.

The sequence model also produces no reason codes, and docs/ARCHITECTURE.md §1 makes an
explanation mandatory on every block. So an escalation always attaches its own reason
code saying which model raised it and why.
"""

from __future__ import annotations

from dataclasses import dataclass

# Above this calibrated sequence score, the sequence model is considered to be raising a
# genuine objection rather than mild disagreement. Chosen as a round, documented
# threshold rather than fitted: fitting it would require a fourth data split, and a
# number presented as optimised when it was picked by hand is exactly what this project
# argues against.
SEQUENCE_CONCERN_THRESHOLD = 0.60

ESCALATION = {"approve": "review", "review": "block", "block": "block"}


@dataclass
class EnsembleOutcome:
    decision: str
    escalated: bool
    sequence_score: float | None
    reason_codes: list[str]


def apply_sequence_opinion(
    decision: str,
    reason_codes: list[str],
    sequence_score: float | None,
    *,
    threshold: float = SEQUENCE_CONCERN_THRESHOLD,
) -> EnsembleOutcome:
    """Let the sequence model escalate a point-risk decision, never relax it."""
    if sequence_score is None or decision == "block" or sequence_score < threshold:
        return EnsembleOutcome(decision, False, sequence_score, reason_codes)

    escalated = ESCALATION[decision]
    return EnsembleOutcome(
        decision=escalated,
        escalated=True,
        sequence_score=sequence_score,
        reason_codes=[
            *reason_codes,
            f"sequence_model_escalation:{decision}_to_{escalated}",
            "anomalous_transaction_sequence_for_account",
        ],
    )
