# Training runbook

Operational companion to `docs/EXPERIMENT_ADVANCED_TRAINING.md` (which records *what* was
tried and found). This is *how to run it* — the exact sequence, what each step writes, and
the checks that keep a result honest.

Commands are PowerShell from the repo root. The venv Python is called explicitly so
nothing depends on which shell has an environment activated.

```powershell
cd C:\Users\kshit\cerberus-risk-engine
$py = ".\.venv\Scripts\python.exe"
```

---

## 0. One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -e .
```

Optional extras, none required to train the shipped model:

```powershell
.\.venv\Scripts\pip.exe install -e ".[tuning]"    # optuna + scikit-optimize
.\.venv\Scripts\pip.exe install -e ".[sequence]"  # torch (GRU)
.\.venv\Scripts\pip.exe install -e ".[gnn]"       # torch + torch-geometric
.\.venv\Scripts\pip.exe install -e ".[llm]"       # anthropic (narration)
```

---

## 1. The standard run

Six scripts, strictly in order — each reads what the previous one wrote. ~10-15 minutes
end to end; feature building and the harness dominate.

```powershell
$py scripts/generate_data.py           # -> data/processed/{transactions,entity_edges}.csv, rings ground truth
$py scripts/detect_rings.py            # -> data/processed/rings_detected.json, reports/ring_detection_report.json
$py scripts/train_baseline.py          # -> models/point_risk_baseline.txt + calibrator, reports/baseline_metrics.json
$py scripts/build_decision_layer.py    # -> reports/decision_layer.json
$py scripts/run_adversarial_harness.py # -> models/point_risk_hardened.txt, reports/adversarial_hardening_report.json
$py scripts/export_dashboard_data.py   # -> dashboard/public/data/*.json
```

One line:

```powershell
$py scripts/generate_data.py; if ($?) { $py scripts/detect_rings.py }; if ($?) { $py scripts/train_baseline.py }; if ($?) { $py scripts/build_decision_layer.py }; if ($?) { $py scripts/run_adversarial_harness.py }; if ($?) { $py scripts/export_dashboard_data.py }
```

**Order is not negotiable.** `detect_rings` before `train_baseline` because the entity
graph feeds `entity_degree`. `build_decision_layer` before the harness because the harness
scores rings against per-segment thresholds. `export_dashboard_data` last, or the
dashboard shows a mix of old and new numbers.

### Variants

```powershell
$py scripts/generate_data.py --calibrate-rate            # real 0.17% fraud rate (see MODEL_CARD)
$py scripts/run_adversarial_harness.py --searcher bayesopt
$py scripts/run_adversarial_harness.py --min-recovery -0.05   # what CI runs
```

---

## 2. Snapshot before you change anything

Without a before, an after is just a number.

```powershell
Copy-Item -Recurse reports reports_snapshot -Force
```

`reports_snapshot/` is gitignored. Take it before each experiment, not once at the start.

---

## 3. The change loop

For every model change, without exception:

```powershell
# 1. snapshot
Copy-Item -Recurse reports reports_snapshot -Force

# 2. make exactly ONE change

# 3. re-run everything downstream of it (see the dependency note below)

# 4. compare
$py -c @"
import json
for f in ['baseline_metrics.json','decision_layer.json']:
    a=json.load(open('reports_snapshot/'+f)); b=json.load(open('reports/'+f))
    print(f)
    for k in ('roc_auc','pr_auc','cost_at_optimal_threshold','overall_savings_pct_vs_global_threshold'):
        if k in a: print(f'  {k:42s} {a[k]:.4f} -> {b[k]:.4f}')
a=json.load(open('reports_snapshot/adversarial_hardening_report.json'))
b=json.load(open('reports/adversarial_hardening_report.json'))
print('adversarial (base / attack / hardened)')
for s in a['strategies']:
    x,y=a['strategies'][s],b['strategies'][s]
    f=lambda d,k: d[k]['combined_score']
    print(f'  {s:20s} {f(x,"baseline_detection"):.2f}/{f(x,"evaded_original_model"):.2f}/{f(x,"evaded_hardened_model"):.2f}'
          f'  ->  {f(y,"baseline_detection"):.2f}/{f(y,"evaded_original_model"):.2f}/{f(y,"evaded_hardened_model"):.2f}')
"@

# 5. tests + lint
$py -m pytest tests/ -q
$py -m ruff check src/ scripts/ tests/

# 6. update every number in the docs that moved
```

**One change at a time.** Two changes and a mixed result tells you nothing about either.

### What to re-run after what

| Changed | Re-run from |
|---|---|
| `data/synthetic_rings.py`, generator config | `generate_data.py` (everything) |
| `features/pipeline.py`, `FEATURE_COLUMNS` | `train_baseline.py` |
| `LGBM_PARAMS`, `point_risk.py` | `train_baseline.py` |
| `decision/cost_matrix.py` | `build_decision_layer.py` |
| `adversarial/*` | `run_adversarial_harness.py` |
| Anything above | always finish with `export_dashboard_data.py` |

---

## 4. Hyperparameter tuning

```powershell
$py scripts/tune_baseline.py --n-trials 40      # ~5-10 min
$py scripts/tune_baseline.py --n-trials 150 --timeout 1800
```

Objective is **cost at the cost-optimal threshold on the calibration split** — not
ROC-AUC, and not the test split. Both choices are deliberate; the script's docstring says
why.

It does not edit anything. Copy the printed `best_params` into `LGBM_PARAMS` in
`src/cerberus/detection/point_risk.py`, then re-run from `train_baseline.py`.

Expect the calibration-split gain to shrink on test — a 4.3% improvement became 0.7%.
That gap is normal and should be reported, not hidden. **If the script says the search
didn't beat the current params, keep them.** A null result is a result.

---

## 5. Optional models

Neither is in the pipeline; both are evaluated side by side and reported.

```powershell
$py scripts/train_sequence.py     # GRU; ~2-4 min CPU
$py scripts/train_gnn_rings.py    # GraphSAGE; ~1 min
```

`train_sequence.py` prints the sequence model, the point-risk model, **and the ensemble**
on the same rows. Only the third number decides whether a second model earns its place.

`train_gnn_rings.py` always runs the `degree >= N` control. **Read that block before the
GNN's own score.** If the control matches, the dataset is too easy and the GNN result says
nothing about the GNN.

---

## 6. Before you commit a model change

- [ ] `pytest tests/ -q` — 28 passing
- [ ] `ruff check src/ scripts/ tests/` — clean
- [ ] Adversarial harness re-run and compared, not just the accuracy metrics
- [ ] A trivial baseline exists for any score above ~0.95
- [ ] `export_dashboard_data.py` re-run so the dashboard matches `reports/`
- [ ] Every number in `README.md`, `MODEL_CARD.md`, `docs/ARCHITECTURE.md`,
      `PROJECT_REPORT.md`, `HANDOFF.md`, `VIDEO_SCRIPT.md` that moved has been updated
- [ ] `cd dashboard; npm run build` if you touched dashboard types or data shape
- [ ] Regressions written down in `docs/EXPERIMENT_ADVANCED_TRAINING.md`, not just fixed

The last two are the ones that get skipped. Docs that disagree with `reports/*.json` are a
bug, and a regression you silently reverted is a lesson you'll relearn.

---

## 7. Reading the numbers

**Where each lives**

| Number | File |
|---|---|
| ROC-AUC, PR-AUC, Brier, ECE | `reports/baseline_metrics.json` |
| Per-segment thresholds and savings | `reports/decision_layer.json` |
| Ring recovery, household FP rate | `reports/ring_detection_report.json` |
| Attack / hardening scores, searcher used | `reports/adversarial_hardening_report.json` |
| Tuning study | `reports/tuning_study.json` |
| Sequence, GNN + degree control | `reports/sequence_metrics.json`, `reports/gnn_ring_metrics.json` |

**What matters, in order**

1. **Adversarial scores** — the differentiator, and the only thing that catches a
   robustness regression hiding behind a better ROC-AUC.
2. **PR-AUC** — with ~2% positives, ROC-AUC flatters. PR-AUC is the honest one.
3. **Cost at the optimal threshold** — the business number.
4. **Calibration (Brier, ECE)** — whether `risk_score` is a probability or just a ranking.
5. **ROC-AUC** — last. Easy to move, easy to move for bad reasons.

**Not comparable across runs:** cost figures across different `base_fraud_rate` settings
(fewer frauds, less cost, no credit due), and adversarial baselines across searchers (each
consumes the RNG differently, so the sandbox rings differ).

---

## 8. Determinism

Everything is seeded by `CERBERUS_RANDOM_SEED` (default 1337).

```powershell
$env:CERBERUS_RANDOM_SEED = "42"; $py scripts/generate_data.py
```

Same seed, same machine, same results. **Re-run any surprising result on two or three
seeds before believing it** — especially the adversarial numbers, where the sandbox rings
are freshly drawn each run and a single run's baseline can swing.
