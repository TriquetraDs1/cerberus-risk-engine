# Handoff — read this first in a new session

**Last updated:** 2026-08-28. **Deadline:** Sept 5, 2026 (Razorpay Track 2 submission).
**Repo:** https://github.com/TriquetraDs1/cerberus-risk-engine (public, clean tree, 7 commits, all pushed).
**Local path:** `C:\Users\kshit\cerberus-risk-engine`

This file exists so a new session (or you, later) can pick up work without re-deriving
context from chat history. If you're an AI assistant starting fresh: read this file,
then `docs/ARCHITECTURE.md`, then you're ready to build — don't re-plan from scratch.

## What Cerberus is

A transaction-risk engine that red-teams itself: point-risk model + graph-based ring
detector + a cost-optimized decision layer, all validated by an adaptive adversarial
harness that attacks the model and proves hardening works. Built for Razorpay's Track 2
hackathon. Defensive-only — see README.md's first paragraph for the scope statement;
repeat that framing in anything you write about this project.

## Status: Days 1-7 of the 10-day plan done, plus 3 items beyond original scope

Everything below is **built, tested, and verified working** — not aspirational.
Beyond the original plan: probability calibration, the case-management workflow, and
the Day 8 A1 narration layer (A2/A3 remain optional stretch).

| Component | Where | Real result |
|---|---|---|
| Synthetic data + rings | `src/cerberus/data/synthetic_rings.py` | 60,565 txns, 25 injected rings, 75 innocent household-sharing pairs, 4 merchant segments |
| Point-risk model | `src/cerberus/detection/point_risk.py` | ROC-AUC 0.80, PR-AUC 0.30 (calibration split costs some AUC — documented trade-off) |
| **Calibration** | `src/cerberus/detection/calibration.py` | Brier 0.0645→0.0164, ECE 0.140→0.003 (isotonic regression) |
| Ring detector (Louvain) | `src/cerberus/detection/ring_detector.py` | 25/25 rings recovered (100%), 9.3% honest FP rate on innocent sharing |
| **Decision layer (Day 4)** | `src/cerberus/decision/cost_matrix.py` | Per-segment cost matrices, 10.7% cheaper than one global threshold |
| **Adversarial harness (Day 5-6)** | `src/cerberus/adversarial/` | 3 adaptive strategies; recall decayed 45-67% under attack, recovered 14-42 points after hardening. Identity rotation barely recovers (honest, reported limitation — Louvain isn't retrainable). |
| **Serving API (Day 7)** | `src/cerberus/serving/app.py` | FastAPI `/score`, `/health`, `/metrics`, `/explain/{id}`, SQLite audit log, demoable graceful degradation |
| **A1 narration (Day 8)** | `src/cerberus/llm/` | 2-3 sentence plain-English summary per decision, in `queue.json` and `/explain`. LLM (Claude) if `ANTHROPIC_API_KEY` set, deterministic template otherwise. Never re-scores. |
| **Dashboard** | `dashboard/` (Next.js) | Review Queue, Ring Network, Adversarial Hardening, System Health — all reading real pipeline output; the drawer shows the A1 Summary |
| **Case management** | `dashboard/app/api/case-actions/`, `lib/caseActions.ts` | Escalate/dismiss rings, mark transactions reviewed, persists server-side |
| CI | `.github/workflows/ci.yml` | Lint → full pipeline → adversarial regression gate → tests (28/28 passing) |
| Docker | `Dockerfile`, `docker-compose.yml` | Two-stage: `pipeline` target and `serving` target |

## How to run everything (from a clean clone)

```bash
cd cerberus-risk-engine
python -m venv .venv && source .venv/Scripts/activate   # or .venv\Scripts\Activate.ps1
pip install -r requirements.txt && pip install -e .
pip install -e ".[llm]"                   # optional: A1 LLM narration (template fallback without it)

python scripts/generate_data.py           # Day 1-2: synthetic data
python scripts/detect_rings.py            # Day 3: Louvain ring detector
python scripts/train_baseline.py          # Day 1-2 + calibration
python scripts/build_decision_layer.py    # Day 4: per-segment routing
python scripts/run_adversarial_harness.py # Day 5-6: attack, harden, prove recovery (~1-2 min)
python scripts/export_dashboard_data.py   # writes dashboard/public/data/*.json

# Dashboard
cd dashboard && npm install && npm run dev   # http://localhost:3000

# Live API (separate terminal, from repo root with venv active)
uvicorn cerberus.serving.app:app --reload   # http://localhost:8000
```

Or: `docker compose run pipeline` then `docker compose up serving`.

Tests: `pytest tests/ -v` (28 tests; `test_serving.py` skips gracefully if the pipeline
hasn't run yet, since it needs a trained model on disk; `test_llm.py` needs neither a
model nor an API key — it pins the deterministic template path).

## Where every number comes from (no mock data anywhere)

`scripts/export_dashboard_data.py` reads the *actual* trained model, calibrator, SHAP
explainer, Louvain output, and adversarial report off disk and writes them to
`dashboard/public/data/*.json`. The dashboard has zero hand-typed numbers. If you
change the model/pipeline, re-run the scripts above in order and the dashboard updates.

## Key design decisions worth knowing before you touch anything

- **Time-based / three-way chronological splits everywhere** (`three_way_split` in
  `point_risk.py`) — never random splits, because fraud rings cluster in time and would
  leak across a random boundary. If you add a new evaluation, follow this pattern.
- **`FEATURE_COLUMNS`** lives in `src/cerberus/features/pipeline.py` — it's the single
  source of truth for the model's input schema (8 base features + 4 one-hot segment
  columns). Every training/scoring path imports it from there. Don't hardcode a
  feature list anywhere else.
- **Reason codes** are shared between the offline export and the live API via
  `src/cerberus/detection/explain.py` — don't duplicate that logic again.
- **The adversarial harness only ever attacks its own sandboxed model.** This is a
  hard, non-negotiable guardrail (see README's first paragraph). Any future evasion
  strategy you add must keep generating fresh synthetic accounts and only scoring
  against locally-loaded model files — never anything that could look like real-world
  evasion tooling.
- **The ring detector's identity-rotation vulnerability is real and intentionally
  unfixed.** Don't "solve" it by quietly tuning Louvain parameters to hide the number —
  the honesty of that limitation is a stated strength of this submission (see
  `docs/ARCHITECTURE.md` panel-pushback section and `MODEL_CARD.md`).
- **The serving API prefers the hardened model** (`models/point_risk_hardened.txt`) if
  it exists, falling back to the baseline. Both are gitignored (regenerable via
  scripts, not committed) — if you clone fresh, you must run the pipeline before the
  API or dashboard export will work.
- **`app/page.tsx` and `app/rings/page.tsx` are `force-dynamic`** — they read mutable
  case-action state from a local JSON file. Don't remove that export or a production
  build will silently serve stale case-action state. `/health` and `/adversarial`
  correctly stay static.
- **A1 narration (`src/cerberus/llm/`) is strictly downstream and non-decisional.** It
  reads a decision that's already been made and writes prose about it. The structured
  `reason_codes` stay authoritative; the LLM text is an extra field. It must never feed
  back into scoring, and it must always have a working templated fallback — the repo,
  the export, and CI all run with no `anthropic` and no `ANTHROPIC_API_KEY`. `anthropic`
  is imported lazily inside `_call_claude`, never at module load.

## What's NOT done (the honest remainder)

- **Day 8 A1 — DONE.** Plain-English reason-code narration is built (`src/cerberus/llm/`,
  `queue.json` `explanation` field, `GET /explain/{id}`, drawer Summary block). What's
  still stretch: **A2** (chargeback dispute drafting) and **A3** (analyst copilot) — see
  `IMPLEMENTATION_ROADMAP.md` Phase 1. Both explicitly optional; do only if the video is
  recorded with days to spare.
- **Day 9-10:** record the submission video, final docs pass, submit by Sept 5. This is
  the actual highest-priority remaining work — not more features.
- **Case-action store is file-based** (`dashboard/data/case_actions.json`, gitignored),
  separate from the Day 7 SQLite audit log. Noted in `lib/caseActions.ts` as a
  known seam to unify later, not a bug.
- No real-data validation of anything (by design — see README's honest-limitations
  section). No auth/RBAC on the dashboard or API. No Kafka/Postgres/Redis — all named
  in `docs/ARCHITECTURE.md`'s Trade-off Analysis as the production path, not built.

## If you're picking this up to keep building, in priority order

1. **Record the video.** You have a working dashboard with a genuinely strong
   before/attack/after chart on `/adversarial` — that's your best 30 seconds.
2. **Re-read `MODEL_CARD.md` and `docs/ARCHITECTURE.md`'s "Anticipated panel pushback"
   section** and rehearse answering those two questions out loud.
3. Only if 1-2 are done with days to spare: A2 (dispute drafting), unify the two audit
   trails, or add a real-data backtest for the ring detector's FP rate (all named as
   "would strengthen this further" in earlier planning, none required).

Do not restart the ML pipeline design, re-litigate synthetic-vs-real data, or rebuild
the dashboard's design system from scratch — all of that is settled and working.
Building "on top" means adding to what's here, not re-deriving it.
