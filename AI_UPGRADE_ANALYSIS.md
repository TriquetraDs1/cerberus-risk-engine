# Cerberus — AI Upgrade Analysis

*Note: `/analyze` isn't a slash command available in this setup, so this is the analysis
written out directly. Scope: "how can we upgrade it and add proper AI, and what's left."*

**Bottom line up front:** the honest "what's left" is the video and the docs pass, not more
model. The one safe AI add — Tier A1, LLM **reason-code narration** — **is now built**
(`src/cerberus/llm/`, `queue.json` `explanation`, `GET /explain/{id}`, drawer Summary; LLM
via Claude with a deterministic template fallback). Everything heavier below is a
post-deadline project.

---

## 1. What "AI" already exists here

Worth being precise, because "add proper AI" implies there's none — there's a fair amount:

| Layer | Technique | Not trivial because |
|---|---|---|
| Point-risk model | Gradient-boosted trees (LightGBM) | Cost-sensitive threshold, not accuracy-max; chronological splits to prevent ring leakage |
| Probability calibration | Isotonic regression on a disjoint split | Makes `risk_score` a real probability; Brier 0.0819 → 0.0163 |
| Ring detection | Entity-link graph + Louvain community detection | Unsupervised structure learning over a ~750-edge entity graph; validated against ground truth |
| Decision layer | Per-segment cost-matrix optimisation | Each segment's FP/FN cost derived from its own data |
| Adversarial hardening | Adaptive local hill-climb over an evasion parameter space, then retrain | This is the differentiator — an optimisation loop searching for model failure |
| Explainability | SHAP attributions → reason codes | Shared between offline export and live API so "why" can't drift |

So the real question isn't "add AI" — it's **"is there an AI component that would strengthen
the submission's story without weakening it."**

---

## 2. The trap: why more model sophistication can *hurt* this submission

The thesis of Cerberus is *honest, explainable, cost-aware, adversarially-tested* fraud
detection built at a scale that's realistic for a solo 10-day build. Its credibility comes
from what it *doesn't* overclaim.

- A **rushed GNN** replacing Louvain would trade an explainable, ground-truth-validated
  component for a harder-to-explain one you can't properly validate in 8 days. `ARCHITECTURE.md`
  already argues against this on the record — reversing it now looks like scope panic.
- A **half-working LLM agent** that sometimes influences decisions breaks the "every
  block/review decision has a deterministic reason code" guarantee, which is a hard
  requirement in `ARCHITECTURE.md` §1.
- Anything that adds a **network dependency** to the core pipeline breaks "a reviewer can
  run this in five minutes offline."

Any AI you add has to respect three constraints: **(a)** it never changes a decision, it
only describes one; **(b)** the repo still runs end-to-end with no API key; **(c)** it's
demoable in the video in under 20 seconds.

---

## 3. Tier A — LLM layer (the "proper AI" add, if any)

This is Day 8 of the original plan. All three options below sit *downstream* of the
deterministic pipeline — they consume `reason_codes`, `cost_basis`, `ring_id`, and the
transaction/graph context that `/score` already produces, and emit text. None of them
feed anything back into scoring.

### A1 — Plain-English reason-code narration  ·  ✅ BUILT

**Status: done.** `src/cerberus/llm/` (`narrate.py`, `templates.py`, `prompts.py`),
`DecisionContext` built from a queue row or an audit row, `narrate_batch` in
`scripts/export_dashboard_data.py` (writes the `explanation` field into `queue.json`),
`GET /explain/{transaction_id}` in `serving/app.py` (echoes `reason_codes` and a
`narration_source` of `"llm"` or `"template"`), and a Summary block in the dashboard
transaction drawer. 9 tests in `tests/test_llm.py`, 2 in `tests/test_serving.py`.
`anthropic` is an optional install (`pip install -e ".[llm]"`); with no key the
deterministic template runs — which is what CI exercises.

The original design, for reference: take the structured output for a flagged transaction
(`reason_codes`, `risk_score`, `decision`, `cost_basis`, ring linkage) and generate a 2–3
sentence analyst-facing explanation such as: *"Blocked at a calibrated risk score of 1.00.
Primary factors: amount anomalous for account, large transaction amount. The account is
linked to flagged ring detected_89. Under the digital subscription cost matrix (a missed
fraud costs about ₹208 versus about ₹18 for a wrong block) the block threshold is set to
0.083."* (That example is real template output from the current `queue.json`.)

- **Where it plugs in:** a new `src/cerberus/llm/narrate.py`, called by
  `scripts/export_dashboard_data.py` (batch, cached to JSON) and optionally exposed as
  `GET /explain/{transaction_id}` on the API.
- **Model:** Claude — `claude-haiku-4-5` is enough for this and keeps latency/cost low;
  `claude-sonnet` if the phrasing quality matters for the video.
- **Guardrails that keep it safe:**
  - The prompt receives only the already-computed artifacts. It's told, explicitly, to
    describe them and never to re-judge risk.
  - `reason_codes_for_row()` in `detection/explain.py` stays the source of truth; the LLM
    text is an *additional* field, never a replacement. If the two disagree, the structured
    codes win and that's visible in the UI.
  - **No key → templated fallback.** If `ANTHROPIC_API_KEY` is unset, emit a deterministic
    template string from the same inputs. The repo still runs for a reviewer.
  - Responses cached by transaction id so the dashboard export is deterministic per run.
- **Why it's the right pick:** it makes the Review Queue visibly better in the demo, it's
  the literal Day 8 deliverable, and it can't break anything because it's read-only text
  over frozen inputs.

### A2 — Chargeback dispute-evidence drafting  ·  ~1.5–2 days  ·  medium risk  ·  high story value

For a `block`/`review` transaction, draft the dispute narrative a real ops team would file:
cite the velocity pattern, the amount anomaly, the ring linkage, the segment cost basis,
in the structure a payment processor expects.

- This is in the original functional requirements (`ARCHITECTURE.md` §1: *"auto-draft
  chargeback dispute evidence"*), so shipping it closes a named gap.
- Same plumbing as A1 (`src/cerberus/llm/`, batch + optional endpoint, templated fallback).
- Higher risk only because the output is longer and format-sensitive, so it needs a couple
  of hand-checked examples in the repo (`reports/sample_disputes/`) to prove quality.
- **Do this only if A1 lands in well under a day.**

### A3 — Analyst "explain this ring" copilot  ·  ~3+ days  ·  higher risk  ·  cut it

Conversational Q&A over a case — the transaction, its graph neighbourhood, the ring's
history. Retrieval over the case context, multi-turn.

- Genuinely useful, genuinely a bigger build: retrieval, chat state, a UI surface,
  prompt-injection surface area from free-text input.
- **Not worth starting with 8 days left.** This is the first thing to build *after*
  submission if you keep developing the project.

---

## 4. Tier B — deeper ML upgrades (post-deadline)

None of these should be touched before 2026-09-05. Listed so the "what's left" picture is
complete and so a future session doesn't rediscover them from scratch.

| Upgrade | What it buys | Why not now |
|---|---|---|
| **GNN ring detector** (GraphSAGE / PyG) as an *alternative* to Louvain, run side by side | A learned, inductive ring signal; a real answer to "Louvain on synthetic data is easy" | Needs far more data and tuning time than 8 days; weakens the explainability story if rushed. Already documented as the production path |
| **Sequence model per account** (small GRU/Transformer over the transaction stream, or richer temporal features) | Directly attacks the slow-ramp weakness — a model that sees the *sequence* catches a stretched burst that 1h-window velocity misses | New model, new training loop, new eval; a week of work to do properly |
| **Stronger adversarial search** — Bayesian optimisation or CMA-ES instead of random hill-climb | A tighter, closer-to-worst-case evasion; lets you claim more than "a local search found this" | Medium effort, and the current honest framing ("randomized hill-climb, can miss a better evasion") is already defensible |
| **Semi-supervised ring hardening** (label propagation / spectral re-clustering after an identity-rotation attack) | Partial recovery on the one strategy that currently doesn't harden | Risky to *claim* a fix for the limitation that's currently a strength precisely because it's unresolved and honest. Only attempt with real rigor |
| **Drift detection wired to auto-retrain** (PSI on incoming features → trigger) | Closes the monitoring loop `ARCHITECTURE.md` §4 describes | Infra, not modelling; low demo value |

---

## 5. Tier C — infrastructure (post-deadline, already documented)

Named in `docs/ARCHITECTURE.md` §5 as the production path, deliberately not built:
Postgres for the audit log, Redis for the entity-graph cache, incremental/streaming graph
updates, horizontal replicas behind a load balancer, auth/RBAC, chargeback-lag-aware label
collection. Unifying the file-based case-action store with the SQLite audit log
(`lib/caseActions.ts`) also lives here — it's the one cleanup that's small enough to do
pre-submission if everything else is done, but it has near-zero panel value.

---

## 6. Recommendation

**Before 2026-09-05, in order:**

1. Record the video + rehearsal (see `PROJECT_REPORT.md` §7 and `VIDEO_SCRIPT.md`). This
   is "what's left."
2. A1 — **done** (§3). Show it in the video: "the analyst sees this sentence, generated
   from the same reason codes the API returns" — then hit `/explain/{id}` and point at the
   matching `reason_codes` in the response.
3. Nothing else before submission.

**After submission, if you keep building:** A2, then the account sequence model (Tier B) —
that's the upgrade that would actually raise the ceiling on detection quality and answer
the slow-ramp weakness with a real architectural change rather than a retrain. Full
sequencing in `IMPLEMENTATION_ROADMAP.md`.

### A1 — as built

```
src/cerberus/llm/
  __init__.py       # exports DecisionContext, narrate_decision, llm_enabled
  narrate.py        # DecisionContext (frozen dataclass), narrate_decision, narrate_batch,
                    #   llm_enabled/narration_source, lazy _call_claude
  templates.py      # render_template(ctx) — deterministic, same inputs, the CI path
  prompts.py        # SYSTEM_PROMPT ("describe, never re-judge") + build_user_prompt

DecisionContext fields (all from a queue row or an audit row — no SHAP, no model):
  transaction_id, decision, risk_score, reason_codes, ring_id, segment, amount,
  fp_cost, fn_cost, block_threshold, review_threshold, actual_label
```

- `scripts/export_dashboard_data.py`: `narrate_batch([DecisionContext.from_queue_row(r)
  for r in queue])`, writes an `explanation` field into `queue.json`. Prints the mode
  (`llm` / `template`).
- `GET /explain/{transaction_id}` in `serving/app.py`: reads the audit row via
  `AuditLog.get`, rebuilds the context from this segment's routing, returns
  `{transaction_id, explanation, reason_codes, narration_source}`.
- Dashboard: a "Summary" block in `TransactionDrawer.tsx`, shown only when
  `explanation` is present; `explanation?: string` added to `QueueTransaction`.
- Tests: `tests/test_llm.py` (9) pins the template path — block/review/approve phrasing,
  no internal codes leaking, `from_queue_row` roundtrip, `use_llm=True` degrading to
  template when `anthropic`/key is absent, batch caching. `tests/test_serving.py` (+2)
  covers `/explain` 200 and 404.
- Packaging: `[project.optional-dependencies] llm = ["anthropic>=0.40"]` in
  `pyproject.toml`; commented in `requirements.txt`. Not installed on CI — CI runs the
  template path, which is the point.
