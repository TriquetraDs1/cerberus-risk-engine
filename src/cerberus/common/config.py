"""Shared paths and constants for the Cerberus pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_CREDITCARD_CSV = DATA_RAW / "creditcard.csv"
SYNTHETIC_TRANSACTIONS_CSV = DATA_PROCESSED / "transactions.csv"
SYNTHETIC_ENTITY_EDGES_CSV = DATA_PROCESSED / "entity_edges.csv"
SYNTHETIC_RINGS_JSON = DATA_PROCESSED / "rings_ground_truth.json"

BASELINE_MODEL_PATH = MODELS_DIR / "point_risk_baseline.txt"

RANDOM_SEED = 1337

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
