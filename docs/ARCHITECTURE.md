# Cerberus — Architecture & System Design

## 1. Requirements

### Functional
- Score each transaction 0–1 for return/chargeback/fraud risk in real time.
- Detect coordinated fraud rings via shared identifiers (device/IP/card/address), not just
  point-in-time scoring.
- Route each transaction: auto-approve / auto-block / human review, based on a
  cost-optimized threshold, not accuracy.
- Explain every flag (feature attribution) for the review-queue analyst.
- Continuously test the detector against simulated evasion strategies and quantify recall
  decay.
- (Stretch) auto-draft chargeback dispute evidence for flagged transactions.

### Non-functional
- Scoring latency: sub-200ms per transaction (real-time gate, not batch).
- Explainability is a hard requirement — every block/review decision needs a reason code.
- Honesty over headline accuracy: every reported metric ships with its false-positive
  cost, not bare precision/recall.
- Auditability: every decision logged, immutable, reconstructible.

### Constraints
- Solo build, ~10 days, submission-grade not production-grade.
- No real transaction data or real fraud rings — everything is public/synthetic.
- Must be demoable in a 5-minute video and runnable by a reviewer in <5 minutes.
- **Hard boundary:** the adversarial component may only attack its own sandboxed model.
  Zero offense-capable artifacts ship in this repo. Non-negotiable — see README.

## 2. High-Level Design

```
Data Layer → Feature Pipeline → Detection Layer (point-risk + ring) → Decision Layer
                                                                              │
                                                                              ▼
                                                       Adversarial Hardening Harness
                                                                              │
                                                                              ▼
                                          Serving: FastAPI + audit log + drift → Dashboard
```

**Data flow:** transaction arrives → features computed (including graph membership) →
both detectors score it → decision layer applies cost-optimal threshold → routed + logged
with reason code → (offline, periodic) harness re-tests the live model against evasion
strategies and flags drift in robustness.

**API contract** (`POST /score`):
```json
{ "transaction_id": "...", "amount": 0, "account_id": "...", "device_id": "...",
  "ip": "...", "card_fingerprint": "...", "timestamp": "..." }
```
→
```json
{
  "transaction_id": "...",
  "risk_score": 0.0,
  "decision": "approve | block | review",
  "reason_codes": ["high_velocity", "shared_device_with_flagged_ring:R482"],
  "ring_id": "R482",
  "cost_basis": { "fp_cost": 0, "fn_cost": 0, "threshold_used": 0 }
}
```
`reason_codes` + `cost_basis` are the artifact that proves "honest metrics" and
"explainability" without a separate slide.

**Storage:** SQLite for the demo (Postgres is the production path — see Trade-offs).
Four tables: `transactions`, `entity_edges`, `decisions` (append-only audit log),
`model_versions`.

## 3. Deep Dive

- **Detection layer split is deliberate:** point-risk (LightGBM) and ring detector
  (Louvain community detection) are independent modules that both write into the decision
  layer. This lets either be retrained without touching the other during the adversarial
  loop, and gives independent, explainable failure modes instead of one opaque joint model.
- **Entity graph caching:** rebuilding community detection per-transaction is wasteful.
  Recompute on a rolling window (every N minutes / M new transactions), cache ring
  membership per entity, look it up at scoring time.
- **Graceful degradation:** if the graph service is stale/down, the point-risk model still
  serves a decision with `ring_check: "unavailable"` — a real code path to demo failing and
  recovering, not a README paragraph.
- **No queue/event system.** Synchronous request/response + a scheduled batch job (graph
  rebuild, adversarial re-test) is honest and sufficient at this scale. Kafka/SQS here
  would read as cargo-culting, not maturity.

## 4. Scale and Reliability (discussion, not a build target)

- At real payment-gateway volume, the point-risk model scales horizontally (stateless
  replicas behind a load balancer). The graph layer is the bottleneck — community
  detection doesn't shard trivially; production path is incremental graph updates or a
  streaming graph library, partitioned by geography/merchant segment.
- Entity-graph cache is the one stateful piece — needs a replicated store (Redis) in
  production; SQLite is fine for the demo.
- Monitoring: population stability index on incoming features (drift before it becomes
  fraud loss), plus rolling recall against confirmed-fraud labels once they arrive.
  Chargebacks land 30–90 days later in reality — that lag is a genuine hard problem, named
  explicitly rather than hidden.

## 5. Trade-off Analysis

| Decision | Chose | Alternative | Why |
|---|---|---|---|
| Two separate detectors vs. one joint model | Separate | Single GNN over all features | Faster to build solo in 10 days, independently explainable, isolated failure modes |
| Synchronous API vs. event-driven | Synchronous + batch job | Kafka/streaming | Matches actual scale of a submission; no infra that adds risk without adding signal |
| Louvain vs. GNN for ring detection | Louvain | GNN | Explainable, fast to validate against synthetic ground truth; a GNN needs far more data/time than 10 days allow, and is a much harder explainability story |
| SQLite vs. Postgres | SQLite | Postgres | Zero setup for a reviewer; Postgres is the noted production path |

**What would change heading toward production:** real labeled fraud data instead of
synthetic rings; a streaming/incremental graph engine; a real human-in-the-loop red-team
process feeding the hardening loop (not just a script); chargeback-lag-aware label
collection; a replicated store for the entity-graph cache.

## 6. 10-day build plan

| Day | Deliverable | Status |
|---|---|---|
| 1–2 | Synthetic data generator with injectable fraud rings + baseline point-risk model. Get a number, any honest number. | ✅ Done — ROC-AUC 0.80, calibrated (Brier 0.0645→0.0164) |
| 3 | Entity-link graph + Louvain on synthetic rings. Confirm it recovers the injected rings. | ✅ Done — 25/25 rings, 100% recovery, 9.3% honest FP rate |
| 4 | Cost matrix + threshold optimization + 3-way routing. | ✅ Done — per-segment, 10.7% cheaper than one global threshold |
| 5–6 | Adversarial harness: 2–3 evasion strategies, measure recall decay, retrain, show recovery. This is the differentiator — protect this time budget above all else. | ✅ Done — 3 adaptive strategies, before/attack/after chart, CI regression gate |
| 7 | FastAPI serving + audit log + drift check. Thin, enterprise-shaped not enterprise-scale. | Not started |
| 8 | Stretch LLM layer (dispute drafting / plain-English reason codes) only if on schedule. | Not started |
| 9 | README + this doc + record the video. | Not started |
| 10 | Submit. | — |

## 7. Anticipated panel pushback (pre-answered)

- *"Your rings are synthetic and your attacker is code you wrote — isn't this just testing
  against yourself?"* Yes — and that's the honest framing, stated explicitly: this
  validates robustness to known evasion classes, not a real-world adversarial guarantee.
  Naming the limitation is the point of "honest metrics."
- *"Louvain on synthetic data is easy — validated on anything real?"* Backtest the ring
  detector's false-positive rate on the non-fraud portion of the Kaggle set (e.g. family
  members sharing a device) — that's the real FP-cost story for the graph layer.
