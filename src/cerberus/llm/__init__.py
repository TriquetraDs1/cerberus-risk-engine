"""A1: plain-English narration of a decision the deterministic pipeline already made.

This layer is downstream of everything. It consumes the structured output of `/score`
and `scripts/export_dashboard_data.py` — decision, calibrated risk score, reason codes,
ring linkage, segment cost basis — and emits 2-3 sentences an analyst can read at a
glance. It never re-scores, never overrides a decision, and never invents a number.

Hard guarantees (see AI_UPGRADE_ANALYSIS.md section 2):
  * The structured `reason_codes` stay authoritative. This text is an *additional* field.
  * No API key -> a deterministic templated sentence from the same inputs. The repo,
    the pipeline, the dashboard export, and CI all run with no key and no network.
  * `anthropic` is an optional dependency, imported lazily. It is not in the core
    requirements and nothing here imports it at module load.
"""

from __future__ import annotations

from cerberus.llm.narrate import DecisionContext, llm_enabled, narrate_decision

__all__ = ["DecisionContext", "llm_enabled", "narrate_decision"]
