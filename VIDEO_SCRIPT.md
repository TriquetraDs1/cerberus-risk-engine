# Cerberus — submission video script

Target length **4:30–5:00** (the reviewer constraint is 5 minutes). Screen recording of
the dashboard + a terminal, your voice over it. Every number below matches
`reports/*.json` as of 2026-08-28 — read them verbatim.

**Before you record:**
- `python scripts/generate_data.py && python scripts/detect_rings.py && python scripts/train_baseline.py && python scripts/build_decision_layer.py && python scripts/run_adversarial_harness.py && python scripts/export_dashboard_data.py`
- `cd dashboard && npm run build && npm run start` (use the production build, not `dev` — no HMR overlay, no recompile pauses)
- Second terminal, repo root, venv active: `uvicorn cerberus.serving.app:app` (no `--reload`)
- Optional: `export ANTHROPIC_API_KEY=...` first if you want the A1 summary to be LLM-written rather than templated. Either is fine to show.
- Zoom the browser to ~110%. Close every other tab. Dark mode or light — pick one and stay.

---

## 0:00 – 0:30 · Thesis

**On screen:** the dashboard home / Review Queue, static.

> "Static fraud thresholds get reverse-engineered by fraud rings within weeks. A model
> that only reports precision and recall on a public dataset has never proven it survives
> an adaptive adversary. Cerberus is a transaction-risk engine that red-teams itself:
> it scores transactions, detects coordinated rings, and then attacks its own detector,
> measures how far recall falls, retrains, and proves the recovery. That loop is the
> submission — not the classifier. Everything you'll see reads from a real pipeline; there
> is no mock data in this dashboard."

---

## 0:30 – 1:30 · The adversarial loop (your strongest 60 seconds)

**On screen:** the `/adversarial` page. Let the before/attack/after chart fill the frame.

> "Three evasion strategies, each found by an adaptive search against the live model, not a
> fixed script. Structuring — split one large payment into many small ones. Identity
> rotation — spread the ring across more devices so the graph never forms a community.
> Slow ramp — stretch the burst so velocity features never fire.
>
> Unattacked, combined detection is 1.00 on all three. Under attack, recall decays 45 to
> 67 percent. For structuring, the point-risk model's catch rate collapses from 100 percent
> to 1 percent.
>
> Then we retrain on what the search found. Structuring recovers to 0.92, slow ramp to
> 0.86. Identity rotation recovers to only 0.47 — and I'll come back to why that one is
> reported, not fixed. This regression check runs in CI on every push."

---

## 1:30 – 2:10 · Point-risk model + calibration

**On screen:** `/health` (System Health) — the reliability diagram.

> "The point-risk model is gradient-boosted trees, ROC-AUC 0.80, on a chronological
> three-way split — never random, because fraud rings cluster in time and would leak
> across a random boundary. The score is isotonic-calibrated: Brier score drops from 0.082
> to 0.016, expected calibration error from 0.19 to 0.002. So `risk_score` is a real
> probability the decision layer can reason about, not just a ranking."

---

## 2:10 – 2:55 · Ring detector + cost-optimised routing

**On screen:** `/rings` (Ring Network), then back to Review Queue; click a `block` row to open the drawer.

> "Coordinated rings are caught with an entity-link graph and Louvain community detection —
> 25 of 25 injected rings recovered, with an honest 9.3 percent false-positive rate on
> innocent households that share a device. That number is reported, not buried.
>
> Routing is cost-optimised per segment, not accuracy-optimised. Each segment's
> false-positive and false-negative costs come from its own transaction data. Segmented
> routing costs 16.5 percent less than one global threshold — and most of that saving is
> in travel, where the ticket size is large and a missed fraud is expensive."

**On screen:** the open drawer — point at Reason codes and Cost basis.

> "Every decision carries its reason codes and its cost basis. That's the 'honest metrics'
> claim as an artifact, not a slide."

---

## 2:55 – 3:35 · Live API + graceful degradation

**On screen:** terminal.

```bash
curl -s -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "transaction_id":"txn_demo","account_id":"acct_1","device_id":"d1","ip":"i1",
  "card_fingerprint":"c1","amount":1899.0,"timestamp":"2026-06-01T02:15:00",
  "segment":"travel_luxury"}' | jq
```

> "The live endpoint serves the hardened model, computes features in real time, returns a
> decision with reason codes, cost basis, and a model version. Sub-200-millisecond."

```bash
curl -s -X POST "localhost:8000/admin/graph-status?status=degraded"
# re-run the /score curl
```

> "Now I take the graph service down. `/score` keeps serving — it drops to `ring_check:
> unavailable` and decides on the point-risk model alone, instead of crashing. That's a
> real code path, not a README paragraph."

```bash
curl -s -X POST "localhost:8000/admin/graph-status?status=fresh"
```

---

## 3:35 – 4:05 · A1 — plain-English narration

**On screen:** the transaction drawer again — the "Summary" block. Then the terminal.

> "Each decision also gets a plain-English summary, built from that decision's own reason
> codes and cost basis. It describes the deterministic output — it never re-scores."

```bash
curl -s localhost:8000/explain/txn_demo | jq
```

> "The endpoint returns the summary and echoes the reason codes it's based on, so you can
> check the prose against the structured output. With no API key it falls back to a
> deterministic template, so the whole repo still runs offline."

---

## 4:05 – 4:40 · Honest limitations + close

**On screen:** talking head, or the MODEL_CARD open.

> "What this doesn't claim. The fraud rings are synthetic and the adversary is code I
> wrote — this validates robustness to known evasion classes, not a real-world guarantee.
> The ring detector's false-positive rate needs a backtest on real non-fraud data.
> Identity rotation attacks Louvain specifically, and Louvain is unsupervised — retraining
> a classifier can't teach a community-detection algorithm anything, so that evasion is
> reported as an open limitation, not patched over.
>
> It also got worse, and I'll tell you why. I added graph features to the point-risk
> model. ROC-AUC went up — to its highest of any configuration I tried — and identity
> rotation collapsed, because the classifier had learned to lean on structure that
> rotation exists to destroy. Aggregate metrics scored that as a clean win. Only the
> harness caught it. That's the argument for having the harness.
>
> Days one through seven of the plan are done, plus calibration, a case-management
> workflow, and the plain-English layer. 28 tests, CI-gated, reproducible from a clean
> clone in five minutes. Thanks for watching."

---

## Rehearse these two out loud (from `docs/ARCHITECTURE.md` §7)

**"Your rings are synthetic and your attacker is code you wrote — isn't this just testing
against yourself?"**
> Yes, and that's the stated framing. It validates robustness to known evasion classes,
> not a real-world adversarial guarantee. Naming that limitation is the point of "honest
> metrics" — a production version needs real labelled fraud and a human red-team process,
> and the architecture doc says exactly that. What the loop demonstrates is the
> *methodology*: build, attack, measure, retrain, prove, gate — which is the lifecycle a
> real fraud team lives in.

**"Louvain on synthetic data is easy — have you validated it on anything real?"**
> Not yet, and that's flagged in the model card and the limitations section. The honest
> next step is a backtest of the ring detector's false-positive rate against the non-fraud
> portion of a real dataset — families sharing a device — because that's the real
> false-positive-cost story for the graph layer. Louvain was chosen over a GNN
> deliberately: explainable, fast to validate against ground truth, and feasible for a
> solo ten-day build. A GNN is named as the production path.

---

## If a shot runs long

Cut in this order: the calibration segment (1:30) down to one sentence; the ring-network
pan; the second `/score` curl in the degradation demo (just show the status flip and
describe it). Never cut the adversarial loop or the limitations — those two carry the
submission.
