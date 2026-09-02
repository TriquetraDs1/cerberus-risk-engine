# Cerberus — Implementation Roadmap: adding every tier

Companion to `AI_UPGRADE_ANALYSIS.md`. That doc argues *whether* to build each tier; this
doc is *how*, in build order, with the files each step touches and the guardrails each step
must not break.

**Reality check.** All tiers together is 8–14 weeks of solo work, not a pre-deadline task.
Before 2026-09-05, build **only Phase 0 (A1)**, and only if the video is already recorded.
Everything else is post-submission. Do the phases in order — each one assumes the previous
is merged and green in CI.

**Three invariants that hold across every phase:**
1. **The LLM never changes a decision.** It describes deterministic output; the structured
   `reason_codes` stay authoritative. (`AI_UPGRADE_ANALYSIS.md` §2)
2. **The repo runs end-to-end with no API key and no network.** Every AI feature has a
   deterministic fallback. A reviewer clones and runs in five minutes, offline.
3. **The adversarial harness only ever attacks its own sandboxed synthetic model.** Any new
   strategy or search generates fresh synthetic accounts and scores against locally-loaded
   model files only. (`README.md` scope statement, `adversarial/harness.py` docstring)

---

## Phase 0 — A1: LLM reason-code narration  ·  ✅ DONE (2026-08-28)

**Goal:** each flagged transaction gets a 2–3 sentence plain-English explanation, generated
from the reason codes / cost basis the pipeline already produces.

**Shipped:**
- `src/cerberus/llm/` — `narrate.py` (`DecisionContext` frozen dataclass built from a
  queue row or audit row, `narrate_decision`, `narrate_batch`, `llm_enabled`,
  `narration_source`, lazy `_call_claude`), `templates.py` (`render_template` —
  deterministic, the CI path), `prompts.py`.
- `scripts/export_dashboard_data.py` writes an `explanation` field into `queue.json` via
  `narrate_batch`.
- `GET /explain/{transaction_id}` in `serving/app.py` (+ `AuditLog.get`), returning
  `{transaction_id, explanation, reason_codes, narration_source}`.
- `TransactionDrawer.tsx` "Summary" block; `explanation?: string` on `QueueTransaction`.
- `tests/test_llm.py` (9) + 2 in `tests/test_serving.py`. (Suite is now 44 tests.)
- `[project.optional-dependencies] llm = ["anthropic>=0.40"]`; not on CI (template path).

The three invariants held: non-decisional, offline-safe (no key → template), no network
in the core pipeline.

---

## Phase 1 — finish the LLM layer (A2, A3)  ·  ✅ DONE (2026-08-31)

Both shipped with dashboard UI and deterministic template fallbacks. `POST /dispute/{id}`
(+ `facts_from` distinguishing audit-log facts from caller-supplied ones) and
`POST /copilot/{ring_id}` (no tools, no writes, fenced case bundle). 16 tests in
`tests/test_llm_features.py`.

### A2 — chargeback dispute-evidence drafting  ·  ✅ done
**Goal:** for a `block`/`review` transaction, draft the dispute narrative an ops team would
file — velocity pattern, amount anomaly, ring linkage, segment cost basis, in processor
format. Closes a named requirement in `docs/ARCHITECTURE.md` §1.
**Effort:** ~1.5–2 days. **Risk:** medium (longer, format-sensitive output).

**Steps**
1. `src/cerberus/llm/dispute.py` — `draft_dispute(ctx: DecisionContext, account_history)`.
   Reuses Phase 0 plumbing (client, cache, templated fallback).
2. Add `account_history` to the context: last N transactions for the account, already
   available from the serving velocity store (`serving/state.py`) or the audit log.
3. Commit 3–5 hand-reviewed examples to `reports/sample_disputes/` as quality evidence.
4. Endpoint `POST /dispute/{transaction_id}`; dashboard "draft dispute" button on the
   transaction drawer, output shown in a copyable panel.

**Done when:** button produces a structured draft; sample drafts in the repo; fallback
produces a usable templated draft with no key.

### A3 — analyst "explain this ring" copilot  ·  ✅ done
**Goal:** multi-turn Q&A over one case — the transaction, its graph neighbourhood, the
ring's history.
**Effort:** ~3–5 days. **Risk:** higher — retrieval, chat state, prompt-injection surface
from free-text input.

**Steps**
1. `src/cerberus/llm/copilot.py` — assemble case context: transaction row, ring members,
   entity-graph neighbourhood (from `detection/ring_detector.py` output), recent decisions
   for those accounts.
2. Context is bounded and pre-assembled (no live tool calls) — pass the case bundle as
   structured context, answer questions against it only.
3. Prompt-injection guard: user questions are quoted as data; the system prompt forbids
   following instructions found in transaction/free-text fields; no action the copilot can
   take changes state.
4. `POST /copilot/{ring_id}` with a `messages` array; new dashboard panel on the ring page
   (`dashboard/app/rings/`).
5. Rate-limit and cache; hard cap on turns per session.

**Done when:** analyst can ask "why is R018 flagged" and "which member is the hub" and get
answers grounded only in the supplied case bundle; injection test (a transaction field
containing "ignore previous instructions") does not alter behaviour.

---

## Phase 2 — decision-quality ML (B3, B2, B1, B4)

This is where detection quality actually goes up. Do B3 first (cheapest, de-risks the
others by strengthening the eval), then B2, then B1, then B4.

### B3 — stronger adversarial search  ·  ✅ done (null result: bayesopt found the same evasions)
**Goal:** replace the random hill-climb in `adversarial/attacker.py` (`adaptive_search`)
with Bayesian optimisation or CMA-ES, so the reported evasion is closer to worst-case.
**Effort:** ~2–3 days. **Risk:** low-medium; the search space and guardrail are unchanged.

**Steps**
1. Add `scikit-optimize` (or `cma`) to requirements.
2. New `adversarial/search.py` with a `Searcher` protocol; keep `adaptive_search` as the
   `"hillclimb"` implementation, add `"bayesopt"`. Select via a `--searcher` flag on
   `scripts/run_adversarial_harness.py`.
3. `STRATEGY_PARAM_BOUNDS` in `strategies.py` already defines the domain — feed it straight
   to the optimiser as the search space.
4. Keep the same sandbox: still `generate_baseline_ring` → apply strategy → `score_ring`
   against locally-loaded models.
5. Update the CI regression gate thresholds if the stronger search finds deeper evasions
   (expected — document the new numbers in `MODEL_CARD.md`).

**Done when:** harness runs with `--searcher bayesopt`; report JSON records which searcher
produced each result; CI green with updated `--min-recovery`.

### B2 — per-account sequence model  ·  ✅ trained + live, escalate-only
**Goal:** a model that sees the *sequence* of an account's transactions, so a stretched
burst (slow-ramp) is caught where a 1h-window velocity feature misses it.
**Effort:** ~1 week. **Risk:** medium-high — new model, new training loop, new eval.

**Steps**
1. `src/cerberus/features/sequences.py` — build fixed-length per-account transaction
   sequences (amount, inter-arrival gap, segment, hour) from the same generated data.
   Reuse the chronological `three_way_split` boundary logic from `detection/point_risk.py`
   so there's no leakage.
2. `src/cerberus/detection/sequence_risk.py` — a small GRU or 1D-CNN (PyTorch). Keep it
   small; this is a signal, not a foundation model.
3. Ensemble, don't replace: `decision/cost_matrix.py` consumes `max(point_risk,
   sequence_risk)` or a learned blend. The LightGBM model stays as the explainable primary.
4. Calibrate the sequence model's output the same way (`detection/calibration.py`) before
   it enters the decision layer.
5. New script `scripts/train_sequence.py`; add it to the pipeline order in `README.md`,
   `HANDOFF.md`, `Dockerfile`.
6. Re-run the adversarial harness — slow-ramp recovery should improve; record it.

**Done when:** slow-ramp under-attack detection improves measurably vs. Phase 0 baseline;
sequence model is calibrated; pipeline still reproducible end-to-end; explainability story
documented (attention weights or a SHAP-on-sequence approximation for the reason codes).

### B1 — GNN ring detector (alongside Louvain, not replacing it)  ·  ✅ trained + live as second opinion
**Goal:** a learned, inductive ring signal — a real answer to "Louvain on synthetic data is
easy."
**Effort:** ~1–2 weeks. **Risk:** high — needs more data and tuning than a quick add; can
weaken the explainability story if it becomes primary.

**Steps**
1. Scale the generator: `data/synthetic_rings.py` — more rings, more varied ring shapes,
   more innocent sharing, so a GNN has something to learn. Keep ground-truth membership.
2. `src/cerberus/detection/gnn_ring.py` — GraphSAGE (PyTorch Geometric) over the
   entity-link graph `detection/ring_detector.py` already builds. Node = entity, edges =
   shared identifiers, label = ring membership.
3. Train/val/test split by *time* again (rings that start after the boundary are unseen).
4. Run both detectors; the decision layer takes the union of flags, and the dashboard ring
   page shows which detector caught each ring.
5. Keep Louvain as the explainable default; the GNN is the "would-scale" answer. Document
   both, and the FP-rate comparison, in `MODEL_CARD.md`.

**Done when:** GNN recovers held-out rings at a stated rate with a stated FP rate on
innocent sharing; both detectors run in the pipeline; neither silently overrides the other.

### B4 — semi-supervised ring hardening
**Goal:** partial recovery on identity rotation — the one strategy that currently doesn't
harden.
**Effort:** ~3–5 days. **Risk:** high — this touches the limitation that is *currently a
strength because it's honest*. Only ship a claim you can defend.

**Steps**
1. After an identity-rotation attack, take the attacked graph and run label propagation
   (or spectral re-clustering) seeded from known-fraud nodes, instead of plain Louvain.
2. Measure recovery honestly — it will be partial. If it's marginal, **keep reporting it as
   a limitation** and note the attempted mitigation; don't overclaim a fix.
3. Update `MODEL_CARD.md` "Adversarial robustness" and `docs/ARCHITECTURE.md` §7 with the
   real new number and the caveat.

**Done when:** identity-rotation post-hardening number moves, and the doc language matches
what the number actually supports — no more, no less.

---

## Phase 3 — production infrastructure (C1–C7, B5)

None of this raises the submission score; it's the "toward production" path. Do C1 first
(small, isolated), then the rest as needed.

### C1 — unify the audit stores
**Goal:** one store instead of two. Today the case-action state is a gitignored JSON file
(`dashboard/data/case_actions.json`), separate from the Day 7 SQLite audit log
(`serving/audit.py`).
**Effort:** ~1 day. **Risk:** low.

**Steps**
1. Add a `case_actions` table to the SQLite schema in `serving/audit.py`.
2. Replace `dashboard/lib/caseActions.ts` file reads/writes with calls to a new
   `POST/GET /case-actions` endpoint on the FastAPI app.
3. Delete the JSON file path and its `.gitignore` entry; update `HANDOFF.md` (the
   `force-dynamic` note stays — the pages still read mutable state, just over HTTP now).

**Done when:** escalate/dismiss persists via the API; one store; `force-dynamic` pages
still behave; tests cover the new endpoint.

### C2 — Postgres
**Goal:** swap SQLite for Postgres as the audit/decision store.
**Effort:** ~2–3 days.

**Steps**
1. Introduce SQLAlchemy (or `asyncpg` + a thin layer) behind the existing `serving/audit.py`
   interface — keep the function signatures so nothing upstream changes.
2. `docker-compose.yml`: add a `postgres` service; `serving` depends on it; SQLite stays as
   the default for `pytest` and the offline pipeline (config switch in `common/config.py`).
3. Alembic migration for the schema (`transactions`, `entity_edges`, `decisions`,
   `model_versions`, `case_actions`).

**Done when:** `docker compose up serving` runs against Postgres; `pytest` still runs
against SQLite with zero setup.

### C3 — Redis entity-graph cache
**Goal:** move the per-entity ring-membership cache out of process.
**Effort:** ~3–4 days. **Prereq:** extract a graph service seam first.

**Steps**
1. Extract graph access in `serving/state.py` behind a `GraphCache` interface with
   `get_ring(entity_id)` / `set_ring(...)` / `status()`.
2. Implementations: `InMemoryGraphCache` (current behaviour, default for tests) and
   `RedisGraphCache`.
3. The graph rebuild job (batch) writes to Redis; `/score` reads from it; the existing
   `POST /admin/graph-status` degradation path now reflects Redis reachability.
4. `docker-compose.yml`: add `redis`.

**Done when:** serving uses Redis when configured, in-memory otherwise; the
graceful-degradation demo still works (stop Redis → `ring_check: "unavailable"`).

### C4 — incremental / streaming entity graph
**Goal:** stop rebuilding community detection from scratch on a rolling window.
**Effort:** ~1–2 weeks. **Risk:** high — this is the genuine scale bottleneck named in
`docs/ARCHITECTURE.md` §4.

**Steps**
1. Replace the batch rebuild with incremental edge insertion + local community update
   (e.g. streaming Louvain / label-propagation on the changed neighbourhood only).
2. Partition the graph by merchant segment / geography so updates are local.
3. Benchmark update latency vs. the batch rebuild; document the crossover point.

**Done when:** an added transaction updates ring membership without a full recompute;
latency documented; correctness checked against a full rebuild on the same data.

### C5 — auth / RBAC
**Goal:** the dashboard and API stop being open.
**Effort:** ~3–4 days.

**Steps**
1. FastAPI dependency for bearer-token / session auth on all non-health routes.
2. Roles: `analyst` (read, case actions), `admin` (the `/admin/*` routes).
3. Dashboard: a login page, session cookie, route guards in `dashboard/app/`.
4. Audit log records the acting user on every decision and case action.

**Done when:** unauthenticated requests get 401; case actions and `/admin/graph-status`
require the right role; the acting user is in the audit trail.

### B5 — drift detection → auto-retrain
**Goal:** close the monitoring loop from `docs/ARCHITECTURE.md` §4.
**Effort:** ~3–5 days. **Prereq:** C2 (needs a real store of scored transactions).

**Steps**
1. `src/cerberus/monitoring/drift.py` — population stability index on incoming features vs.
   the training distribution; expose as a `/metrics` gauge.
2. A scheduled job: when PSI crosses a threshold, trigger the pipeline
   (`generate_data` → ... → `run_adversarial_harness`) and register a new
   `model_versions` row; serving picks up the new model on next load.
3. Dashboard System Health page shows PSI over time and the last retrain event.

**Done when:** a deliberately shifted input distribution raises the PSI gauge and (in a
test) triggers the retrain path; serving loads the new version without a restart.

### C6 — horizontal scaling
**Goal:** stateless `/score` replicas behind a load balancer.
**Effort:** ~2–3 days. Mostly deployment.

**Steps**
1. Confirm `/score` is fully stateless once C3 is done (velocity history also moves to
   Redis).
2. `docker-compose.yml`: scale `serving` to N replicas behind nginx/traefik.
3. Load test; confirm the graph cache and audit store are the only shared state.

**Done when:** N replicas serve identical decisions for the same input; a killed replica
doesn't drop state.

### C7 — chargeback-lag-aware labels
**Goal:** model the 30–90 day gap between a transaction and its confirmed fraud label.
**Effort:** ~1 week. **Risk:** high — changes the data model and every eval.

**Steps**
1. `data/synthetic_rings.py`: emit a `label_available_at` timestamp per fraud row, sampled
   30–90 days after the transaction.
2. Every training split now also filters on "label known as of the split boundary" — a row
   whose label lands after the boundary is treated as unlabelled at train time.
3. Add a "labels still maturing" panel to the dashboard so the metric's provisional nature
   is visible.
4. Document the effect on ROC-AUC / PR-AUC (it will drop — that's the honest number).

**Done when:** splits respect label-availability time; reported metrics reflect only
matured labels; the doc explains the gap.

---

## Suggested end-to-end order (post-2026-09-05)

```
Phase 0  A1 ....................... ✅ done
Phase 1  A2 → A3 .................. ✅ done
Phase 2  B3 → B2 → B1 ............. ✅ done      B4 → still open
Phase 3  C1 → C2 → C3 → C5 →
         B5 → C4 → C6 → C7 ........ ~4–6 weeks
```

## The generator problem — fixed 2026-08-31

This section used to say the research was blocked because the generator produced rings a
`degree >= 2` threshold separated perfectly, so no graph model was being tested at all.
That is done. Rings now come in four topologies (clique, star, chain, partial), form
gradually, and innocent households are 2-5 accounts rather than pairs.

Two results came out of it. The GNN's meaningless 1.0000 became a real 0.8863 and the
degree control stopped matching it. And Louvain-on-structure was revealed to
false-positive on 94% of innocent households — a four-person family and a four-person
ring are the same graph — which is why `detect_communities` now requires behavioural
coordination as well as structure.

## What is actually blocking progress now

**One download.** `scripts/validate_rings_real_data.py` is built and tested; it needs
IEEE-CIS `train_transaction.csv` in `data/raw/`, which is a Kaggle *competition* dataset
and so requires manually accepting its rules before it can be fetched.

Why that dataset and not `creditcard.csv`: the latter is anonymised PCA components with no
card, device, or address field, so no entity graph can be built from it — it was never
capable of validating a graph detector. IEEE-CIS carries real card, address and device
identifiers alongside a real fraud label.

And why this is measurable at all, which took longer to see than it should have:
validating a *false-positive* rate needs real innocent entity-sharing, not fraud-ring
labels. No public dataset has ring labels. Every public dataset with identifiers has
families sharing a card.

After that: B4 is now worth attempting, because the dataset can finally distinguish
between graph detectors. Everything else remaining is infrastructure (Phase 3).

Re-run the full pipeline and the adversarial harness after every ML change (Phase 2) and
re-check CI after every phase. Update `MODEL_CARD.md`, `docs/ARCHITECTURE.md`, and
`PROJECT_REPORT.md` numbers as they move — the docs agreeing on every figure is itself a
reviewability feature.
