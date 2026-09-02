"""Shared configuration for the Cerberus pipeline.

Paths are fixed relative to the repo root. Tunable parameters (cost ratios, random
seed) are exposed as environment-overridable settings via pydantic — the standard
pattern for anything a deployment might need to change without a code edit, e.g.
tuning the cost matrix per merchant segment without redeploying.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_CREDITCARD_CSV = DATA_RAW / "creditcard.csv"

# IEEE-CIS Fraud Detection. Unlike creditcard.csv (anonymised PCA components with no
# identifiers), this carries real card, address, and device fields — which is what makes
# it the only public dataset that can validate the ring detector's false-positive rate
# against genuine innocent entity-sharing. See scripts/validate_rings_real_data.py.
IEEE_TRANSACTION_CSV = DATA_RAW / "train_transaction.csv"
IEEE_IDENTITY_CSV = DATA_RAW / "train_identity.csv"
SYNTHETIC_TRANSACTIONS_CSV = DATA_PROCESSED / "transactions.csv"
SYNTHETIC_ENTITY_EDGES_CSV = DATA_PROCESSED / "entity_edges.csv"
SYNTHETIC_RINGS_JSON = DATA_PROCESSED / "rings_ground_truth.json"
SYNTHETIC_HOUSEHOLD_PAIRS_JSON = DATA_PROCESSED / "household_pairs_ground_truth.json"
DETECTED_RINGS_JSON = DATA_PROCESSED / "rings_detected.json"
RING_DETECTION_REPORT_JSON = REPORTS_DIR / "ring_detection_report.json"

BASELINE_MODEL_PATH = MODELS_DIR / "point_risk_baseline.txt"
CALIBRATOR_PATH = MODELS_DIR / "point_risk_calibrator.joblib"
BASELINE_METRICS_JSON = REPORTS_DIR / "baseline_metrics.json"

# The Next.js dashboard reads its data as static JSON from its own public/ dir —
# no live backend required for the demo. See scripts/export_dashboard_data.py.
DASHBOARD_DATA_DIR = ROOT / "dashboard" / "public" / "data"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Env-overridable knobs. Override any of these with a `CERBERUS_` prefixed
    env var, e.g. `CERBERUS_FN_COST=80` to reflect a merchant with costlier fraud.
    """

    model_config = SettingsConfigDict(env_prefix="CERBERUS_", env_file=".env", extra="ignore")

    random_seed: int = 1337

    # Placeholder cost ratio: refine with real numbers once a merchant/segment
    # cost study exists. See docs/ARCHITECTURE.md, Day 4 decision layer.
    fp_cost: float = 5.0
    fn_cost: float = 50.0

    # Louvain resolution parameter — higher finds smaller, tighter communities.
    louvain_resolution: float = 1.0


settings = Settings()
