# Experiment: advanced training (branch `advanced-training`)

**Date:** 2026-08-28 · **Baseline:** `main` @ `382bc43`, snapshot in `reports_baseline_snapshot/`
**Status:** complete for Steps 1–3; Steps 4–6 outstanding (see bottom).

Two changes, measured independently and then together: cost-objective hyperparameter
tuning, and a richer feature set. One of the two produced a **regression** that is
reported here rather than buried — the finding is the more interesting half of the result.

---

## Step 2 — Optuna tuning against cost, not AUC

`scripts/tune_baseline.py`. 40 TPE trials. Objective: **total expected cost at the
cost-optimal threshold**, evaluated on the held-out calibration split with scores
calibrated first.

Tuning for ROC-AUC would have optimised the exact metric this project argues is the wrong
target (`docs/ARCHITECTURE.md` §1). Tuning on the test split would have leaked, since
hyperparameter selection is a fitting decision.

| | Cost on calibration split |
|---|---|
| Previous hand-set params | 7,840 |
| Best of 40 trials | **7,505** (−4.3%) |

On the untouched test split the gain shrank to ~0.7% — reported as measured. Best params
are now `LGBM_PARAMS` in `src/cerberus/detection/point_risk.py`; the study is written to
`reports/tuning_study.json`.

---

## Step 3 — Richer features

Added to `src/cerberus/features/pipeline.py`:

- **Multi-window velocity** — counts and amounts over `1h`, `24h`, `7d` (was `1h` only).
  Aimed squarely at slow-ramp evasion, which works by stretching a burst past a single
  short window.
- **`amount_vs_trailing_mean`** — amount over the account's expanding, `shift(1)` mean.
  Trailing-only by construction, so the live API can reproduce it.
- **`hours_since_last_txn`** — gap since the account's previous transaction.
- **`hour_sin` / `hour_cos`** — cyclical hour encoding, so hour 23 and hour 0 are adjacent.
- **`shared_entity_strength`, `component_size`** — graph features. **Computed but
  deliberately excluded from `FEATURE_COLUMNS`** — see the regression below.

`src/cerberus/serving/app.py` and `serving/state.py` were extended to compute every new
feature at request time from in-process history, with the same first-transaction fallbacks
the offline pipeline uses, so training and serving stay in agreement.

---

## The regression: graph features made identity rotation strictly worse

Three configurations, all measured with the same harness budget (5 restarts × 15 steps):

| Strategy | | Baseline | Under attack | After hardening |
|---|---|---|---|---|
| **structuring** | original | 1.00 | 0.51 | 0.92 |
| | + graph features | 1.00 | 0.50 | **1.00** |
| | **shipped** (no graph feats) | 1.00 | 0.50 | **1.00** |
| **identity_rotation** | original | 1.00 | **0.33** | 0.47 |
| | + graph features | 1.00 | **0.00** | 0.50 |
| | **shipped** (no graph feats) | 1.00 | **0.04** | 0.50 |
| **slow_ramp** | original | 1.00 | 0.55 | 0.86 |
| | + graph features | **0.84** | 0.78 | 1.00 |
| | **shipped** (no graph feats) | 1.00 | **0.80** | **1.00** |

**What happened.** Feeding `component_size` and `shared_entity_strength` to the *point-risk*
model let it lean on graph structure — and identity rotation is precisely the attack that
destroys graph structure. The classifier's independent signal collapsed: under attack it
went from catching 67% of rotated rings to catching none. It also depressed the unattacked
slow-ramp baseline to 0.84.

Removing those two columns from `FEATURE_COLUMNS` (keeping the functions, since the ring
detector and a future GNN want them) restored the slow-ramp baseline to 1.00 and recovered
identity rotation to 0.04.

**The general lesson, which is the part worth saying out loud:** a feature that improves
aggregate accuracy can *reduce* adversarial robustness by concentrating the model's
reliance on a signal an attacker controls. Aggregate metrics alone would have shown this
change as a clean win — ROC-AUC was *highest* (0.8206) in the configuration with the worst
adversarial behaviour. Only the harness caught it. That is the argument for having the
harness.

---

## Net result of the shipped configuration

| Metric | Before | After | |
|---|---|---|---|
| ROC-AUC | 0.8006 | 0.8153 | ↑ |
| PR-AUC | 0.3019 | **0.3141** | ↑ (best of all three configs) |
| Brier (calibrated) | 0.0164 | 0.0163 | ~ |
| ECE (calibrated) | 0.0032 | 0.0023 | ↑ |
| Cost at optimal threshold | 8,850 | 8,780 | ↑ |
| Segmented routing savings | 10.7% | **16.5%** | ↑ |
| Slow-ramp decay under attack | 0.45 | **0.20** | ↑ (the intended win) |
| Structuring, after hardening | 0.92 | **1.00** | ↑ |
| **Identity rotation under attack** | **0.33** | **0.04** | **↓ regression** |

Eight metrics improved; one regressed. The regression is on the strategy that was already
this project's documented unfixed limitation, and it is now *more* unfixed than before.

**Recommendation:** merge only if the identity-rotation regression is stated plainly
wherever the adversarial numbers appear (`MODEL_CARD.md`, `README.md`, the video). If
that framing is not wanted for the submission, `main` @ `382bc43` remains the stable
alternative and this branch stands as a documented experiment. Do not merge and quietly
keep the old identity-rotation number — the two do not go together.

---

---

## Step 1 — calibrating against real data (Kaggle `creditcard.csv`, 284,807 rows)

`data/loader.py` now measures the reference set and can anchor the generator to it.
Two things it found, and one it broke.

### The reference overturned two hand-picked assumptions

| | Assumed | Measured |
|---|---|---|
| Base fraud rate | 1.8% | **0.17%** — an order of magnitude rarer |
| Fraud amount vs. legitimate | higher mean (μ 6.5 vs 5.2) | **lower** mean, far wider spread (μ 2.65/σ 2.56 vs μ 3.01/σ 1.87) |

The second is the more interesting error. The generator assumed fraud means big-ticket
transactions. Real card fraud is full of *small* card-testing charges — the distribution
is lower-centred and much heavier-tailed, not simply shifted up.

### Amounts are not transferable; the rate is

Importing the reference's amount parameters wholesale made things measurably worse:
per-segment routing savings collapsed to **0.0%** and the unattacked adversarial
baselines fell to 0.50. The cause is a units mismatch — the reference is European card
data with a ~$22 median, while this generator models an INR stream whose ring-structuring
threshold is ₹2000. Transferred amounts produced rings priced ~100× the legitimate
median: internally incoherent data on which segments stopped differing in any way a cost
matrix could exploit.

So `calibrate_config_to_reference` takes the **rate** (dimensionless, comparable across
markets) and leaves amounts synthetic, with `calibrate_amounts=True` available for a
reference that actually shares a currency.

### Rate-only calibration: better model, thinner demo

`python scripts/generate_data.py --calibrate-rate`

| Metric | Assumed 1.8% | Real 0.17% | |
|---|---|---|---|
| ROC-AUC | 0.8153 | **0.8577** | ↑ |
| PR-AUC | 0.3141 | **0.5091** | ↑ |
| ECE (calibrated) | 0.0023 | 0.0012 | ↑ |
| Segmented routing savings | 16.5% | **9.0%** | ↓ |
| Review-tier volume | 772 | **0** | ↓ |
| Adversarial baselines | ~1.00 | 0.64–0.89 | ↓ |

The model gets *better* on the truthful distribution — notably PR-AUC, the metric that
matters under class imbalance. What thins out is everything downstream that needs
positives to work with: at a tenth the fraud rate, 60k transactions hold ~100 base-fraud
rows, the review tier empties, and the harness's sandbox baselines weaken.

Those are **small-sample artefacts, not findings about the method.** The statistically
correct response is more transactions, not a fatter fraud rate — unspent pipeline time
this build hasn't paid yet. Until then the real rate is behind `--calibrate-rate`,
both configurations reproduce, and the numbers for both are in this table. Cite whichever
you run; just say which one it was.

---

## Step 4 (B3) — Bayesian-optimisation searcher

`src/cerberus/adversarial/search.py`, selected with
`python scripts/run_adversarial_harness.py --searcher bayesopt`. Gaussian-process
optimisation over the same `STRATEGY_PARAM_BOUNDS`, same evaluation budget, same sandbox
guardrail. The chosen searcher is recorded in the report JSON.

**It found essentially the same evasions as the hill-climb** — structuring 0.50 under
both, identity rotation 0.04 under both. That is a null result and worth keeping:
`attacker.py` has always *asserted* that this detection surface is "small and fairly
smooth", such that a simple auditable search finds what a heavier method would. That
claim is now tested rather than asserted, by the heavier method agreeing with it.

Caveat on comparing the two: each searcher consumes the shared RNG differently, so the
freshly-drawn sandbox rings differ between runs and the unattacked baselines move
(1.00 vs 0.57 for identity rotation across the two runs above). Cross-searcher numbers
are indicative, not paired. A paired comparison would need the ring draw seeded
independently of the search — worth doing before quoting a head-to-head anywhere.

The default stays `hillclimb`: no extra dependency, and now with evidence it is not
leaving anything on the table.

---

## Step 5 (B2) — per-account sequence model

`features/sequences.py`, `detection/sequence_risk.py`, `scripts/train_sequence.py`.
A 2-layer GRU (48 hidden, ~20k params) over the 8 transactions ending at the one being
scored: log amount, log inter-arrival gap, cyclical hour, one-hot segment. Strictly
causal, left-padded, split on the same chronological boundaries.

| Held-out | ROC-AUC | PR-AUC |
|---|---|---|
| Point-risk (LightGBM) | 0.8153 | 0.3141 |
| Sequence (GRU) | **0.8371** | 0.3011 |
| Ensemble (70/30) | 0.8272 | **0.3341** |

The GRU beats the booster on ROC-AUC and loses on PR-AUC — they are finding different
things, which is the case for keeping both. The ensemble beats point-risk alone on
PR-AUC (0.3341 vs 0.3141), which is the only comparison that justifies shipping a second
model at all.

**It stays a second opinion at weight 0.3, not a co-equal vote,** because it produces no
reason codes. `docs/ARCHITECTURE.md` §1 makes an explanation mandatory on every block; a
model that scores well but cannot say why is not allowed to overrule one that can. The
weight is a chosen constant, not a fitted one — fitting it would need a fourth split.

## Step 6 (B1) — GNN ring detector, and why its perfect score means the opposite of what it looks like

`detection/gnn_ring.py`, `scripts/train_gnn_rings.py`. GraphSAGE, 2 layers, 5 node
features, temporal node split by each account's first transaction.

Held-out **ROC-AUC 1.0000, PR-AUC 1.0000**. 25/25 rings recovered, household
false-positive rate 1.3% against Louvain's 9.3%.

A perfect held-out score is a reason for suspicion, so the script now always runs a
one-line control: **flag every account with `degree >= 2`.**

| | Recovery | Household FP |
|---|---|---|
| Louvain | 100% | 9.3% |
| GraphSAGE | 100% | 1.3% |
| **`degree >= 2`** | **100%** | **1.3%** |

The trivial baseline matches the GNN exactly. Message passing contributed nothing: the
injected rings are dense cliques (degree 5–11) and the innocent households are single
links (degree 1), so one integer comparison separates them perfectly.

**The correct reading is that this measures the synthetic graph's easiness, not the
GNN's power** — and by extension it is evidence *for* the panel's pushback ("Louvain on
synthetic data is easy"), not against it. Reporting the 1.0000 as a GNN achievement
would be the single most misleading number this project could publish. The control runs
on every invocation and its output is written into `reports/gnn_ring_metrics.json`
precisely so that number can never be quoted alone.

What would make the comparison meaningful: ring shapes that are not uniformly dense
(chains, stars, partial overlaps), innocent clusters larger than pairs, and an inductive
split where a held-out account's neighbours are also held out. Until the generator
produces graphs a degree threshold cannot solve, no graph model — learned or
unsupervised — is being tested by this dataset.

## Not done on this branch

- Regenerating the synthetic graph with harder, more varied ring topologies, which is
  the prerequisite for Step 6's comparison to mean anything.
- Wiring the sequence model into the live decision layer (it is trained, calibrated, and
  saved, but `decision/cost_matrix.py` still consumes point-risk alone).
- A paired searcher comparison with the sandbox ring draw seeded independently of the
  search (see Step 4's caveat).

## Reproducing

```bash
python scripts/generate_data.py && python scripts/detect_rings.py
python scripts/tune_baseline.py --n-trials 40      # optional, writes reports/tuning_study.json
python scripts/train_baseline.py && python scripts/build_decision_layer.py
python scripts/run_adversarial_harness.py && python scripts/export_dashboard_data.py
```
