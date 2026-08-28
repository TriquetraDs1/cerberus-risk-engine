# Deploying Cerberus — frontend, backend, and the LLM layer

## What the three pieces are, and how coupled they are today

| Piece | What it is | Needs at runtime |
|---|---|---|
| **Frontend** | `dashboard/` — Next.js analyst console | The static JSON in `dashboard/public/data/` (committed). Nothing else. |
| **Backend** | `cerberus.serving.app` — FastAPI `/score`, `/explain/{id}`, `/health`, `/metrics`, `/audit/recent`, `/admin/graph-status` | Model artifacts on disk (`models/`, `reports/decision_layer.json`, `data/processed/*.csv`). SQLite audit log. |
| **AI / LLM** | `cerberus.llm` — decision narration | Nothing to run the template path. `ANTHROPIC_API_KEY` for LLM-written text. |

**Important:** the frontend does **not** call the backend right now. The four dashboard
pages render *analytical export* data (queue, ring graph, calibration, adversarial report)
produced offline by `scripts/export_dashboard_data.py`. The `/score` + `/explain` API is a
separate live service. They can be deployed independently (Path A) or wired together
(Path B).

---

## Path A — deploy the three pieces live (recommended before 2026-09-05, ~30 min)

Keeps the current architecture. Gets you a clickable dashboard URL and a live API URL for
the video. Loosely coupled, exactly as designed.

### A1. Frontend → Vercel

The dashboard has all the JSON it needs committed in `public/data/`, so it deploys with
**zero backend and zero env vars**.

- **Dashboard UI:** vercel.com → Add New → Project → import `TriquetraDs1/cerberus-risk-engine`
  → **set Root Directory to `dashboard`** → framework auto-detects Next.js → Deploy.
- **Or CLI:** `cd dashboard && npx vercel` then `npx vercel --prod`.

`/` and `/rings` are `force-dynamic`, so they run as serverless functions (Vercel handles
this automatically — a plain static export would break them). The `/api/case-actions`
route writes to a local file, which is **ephemeral on serverless** — escalate/dismiss will
appear to work within a session but won't persist across cold starts. Fixing that is
Path B / roadmap C1.

### A2. Backend → Render (or Railway / Fly)

The API needs model artifacts. The `serving-standalone` Docker target (the Dockerfile's
last stage) **bakes them in at build time** — training runs on the build machine, and the
running container only loads a ~1 MB booster, so it stays well under a 512 MB free-tier
RAM limit and boots instantly.

- **Render Blueprint:** `render.yaml` is in the repo. Render → New → Blueprint → pick this
  repo → it creates a Docker web service on the free plan with a `/health` check.
- **Render manual:** New → Web Service → this repo → Runtime **Docker** → deploy.
- **Then set env vars** (Render → your service → Environment):
  - `ANTHROPIC_API_KEY` — enables LLM narration on `/explain`. Without it, the template runs.
  - `CERBERUS_CORS_ORIGINS` — set this to your Vercel URL (e.g. `https://cerberus-xyz.vercel.app`)
    only if you do Path B. Leave unset for Path A.

**Free-tier gotchas:**
- The service spins down after ~15 min idle; next request has a ~50 s cold start (just the
  container, no pipeline). **Hit the URL once a few minutes before you present.**
- The SQLite audit log lives in the container's ephemeral filesystem — it resets on every
  deploy/restart. Fine for a demo. Persistence = roadmap C2 (Postgres) or a Fly.io volume.
- The baked image serves the **baseline** model (the adversarial harness is skipped at
  build). `/score` and `/explain` need nothing more. To serve the hardened model instead,
  add `run_adversarial_harness.py --min-recovery -1.0` to that `RUN` in the Dockerfile.

### Alternative: Hugging Face Spaces (16 GB RAM free, no card)

Best free option for the ML stack. Create a **Docker** Space, then:
- Point it at this repo, or add the repo as a remote and push.
- In the Space's `README.md` YAML frontmatter set `app_port: 8000` (Spaces serve on 7860
  by default; the API listens on `${PORT:-8000}`), or set a `PORT` space variable to 7860.
- Space **Settings → Variables and secrets**: add `ANTHROPIC_API_KEY` as a secret.
- The Space builds the Dockerfile's last stage (`serving-standalone`) automatically.
- Free Spaces sleep after 48 h idle (much longer than Render's 15 min).

**Railway:** Deploy from repo → reads the `Dockerfile` → add the env vars. Free trial
credit only, then paid. **Fly.io:** `fly launch`, `fly deploy`, `fly secrets set
ANTHROPIC_API_KEY=…`; add a volume + `[mounts]` for a persistent `data/`. Also trial
credit, then paid.

### A3. LLM layer

Nothing to deploy — it's in the backend. Setting `ANTHROPIC_API_KEY` on the backend host
(A2) is the whole step. `GET /explain/{id}` then returns `narration_source: "llm"`.

To also get LLM-written text into the dashboard's static `queue.json` (Path A keeps it
static): run the export locally with the key set, commit the result:
```bash
export ANTHROPIC_API_KEY=…
pip install -e ".[llm]"
python scripts/export_dashboard_data.py     # prints "(llm mode)"
git add dashboard/public/data/queue.json && git commit -m "queue: LLM-written explanations"
```

### A4. Verify

```bash
curl https://<your-render-url>/health
curl -X POST https://<your-render-url>/score -H 'content-type: application/json' -d '{
  "transaction_id":"t1","account_id":"a1","device_id":"d1","ip":"i1",
  "card_fingerprint":"c1","amount":1899,"timestamp":"2026-06-01T02:15:00","segment":"travel_luxury"}'
curl https://<your-render-url>/explain/t1
```
Open the Vercel URL, click a `block` row, confirm the drawer + Summary render.

---

## Path B — wire them into one full-stack app (post-deadline)

This is the "real" integrated system. It's days of work and most of it is already scoped
in `IMPLEMENTATION_ROADMAP.md` (Phase 3, C1–C2). Do it after the submission.

### B1. Let the browser call the API
- Backend: set `CERBERUS_CORS_ORIGINS` to the dashboard origin — the `CORSMiddleware` is
  already wired in `serving/app.py`, env-gated (off by default).
- Frontend: add `NEXT_PUBLIC_API_BASE_URL` (Vercel env var) → your Render URL. New
  `dashboard/lib/api.ts` with `fetch` helpers.

### B2. Add live surfaces (don't convert the analytical pages)
The queue / ring-graph / calibration / adversarial pages render pipeline *analysis* — they
stay export-driven. Integration means **adding** live surfaces next to them:
- A "Score a transaction" form component → `POST ${API}/score` → render decision, reason
  codes, cost basis, and the `/explain` summary.
- An "Audit log" view → `GET ${API}/audit/recent`.
- Wire the drawer's Summary to call `${API}/explain/{id}` live instead of reading the
  baked `explanation` field.

### B3. Unify state and persistence
- **Case actions** (roadmap **C1**): replace the Next.js local-file route
  (`dashboard/app/api/case-actions/`, `lib/caseActions.ts`) with backend endpoints and a
  `case_actions` table in the audit DB. Removes the "ephemeral on serverless" problem.
- **Persistence** (roadmap **C2**): swap SQLite for Postgres behind `serving/audit.py`
  (Render and Railway both offer a managed Postgres add-on), or attach a Fly.io volume to
  keep the SQLite file.

### B4. One-container option
If you'd rather ship frontend + backend as a single deployable: `next build && next
export` won't work (the `force-dynamic` pages), so run the Next.js server and FastAPI
behind one reverse proxy (nginx/Caddy) in a compose stack, or use Next.js
`rewrites` to proxy `/api/*` to the FastAPI service. This is roadmap C6 territory and only
worth it at real volume.

---

## TL;DR

- **Before Sept 5:** Path A. Vercel for `dashboard/` (root dir = `dashboard`, no env),
  Render Blueprint (`render.yaml`) for the API, set `ANTHROPIC_API_KEY`. Warm the API
  before you present.
- **After Sept 5:** Path B, following `IMPLEMENTATION_ROADMAP.md` Phase 3.
