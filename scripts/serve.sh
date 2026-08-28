#!/bin/sh
# Self-bootstrapping entrypoint for a container host (Render / Railway / Fly / a plain
# `docker run`) that has no mounted volumes. If model artifacts aren't present, run the
# offline pipeline once to produce them, then start the API.
#
# Local dev doesn't need this — use `uvicorn cerberus.serving.app:app --reload` directly,
# or `docker compose run pipeline && docker compose up serving`.
set -e

if [ ! -f models/point_risk_baseline.txt ]; then
  echo "serve.sh: no model artifacts found — running the pipeline once (~2-3 min)..."
  python scripts/generate_data.py
  python scripts/detect_rings.py
  python scripts/train_baseline.py
  python scripts/build_decision_layer.py
  # The hardened model is preferred by the API but not required; don't fail startup if
  # the harness has a bad run on the host.
  python scripts/run_adversarial_harness.py --min-recovery -1.0 || \
    echo "serve.sh: adversarial harness failed — serving the baseline model instead."
fi

# $PORT is set by most hosts (Render injects it). Fall back to 7860, which is what
# Hugging Face Spaces expects, so the same image runs on both without extra config.
exec uvicorn cerberus.serving.app:app --host 0.0.0.0 --port "${PORT:-7860}"
