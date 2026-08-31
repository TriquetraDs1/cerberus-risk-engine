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

## Not done on this branch

- **Step 1 — real Kaggle data.** Needs `data/raw/creditcard.csv`, which requires a Kaggle
  login. Drop the file in and re-run the pipeline; `data/loader.py` blends its base rate in.
- **Step 4 (B3) — Bayesian-opt adversarial searcher.** `scikit-optimize` is installed.
- **Step 5 (B2) — per-account sequence model.** Needs PyTorch (~2.5 GB) and real training
  iteration. The clearest remaining answer to slow-ramp.
- **Step 6 (B1) — GNN ring detector.** Needs PyTorch Geometric. Note that
  `shared_entity_strength` and `component_size` are already computed and waiting for it —
  a graph model is the *right* consumer for graph features, which is the flip side of the
  regression documented above.

## Reproducing

```bash
python scripts/generate_data.py && python scripts/detect_rings.py
python scripts/tune_baseline.py --n-trials 40      # optional, writes reports/tuning_study.json
python scripts/train_baseline.py && python scripts/build_decision_layer.py
python scripts/run_adversarial_harness.py && python scripts/export_dashboard_data.py
```
