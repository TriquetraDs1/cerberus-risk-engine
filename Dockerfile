# Build targets from one image:
#   docker build --target pipeline           -t cerberus-pipeline .
#   docker build --target serving            -t cerberus-serving .
#   docker build --target serving-standalone -t cerberus-serving-standalone .   (also the default — last stage)
#
# `pipeline` reproduces every number in the README without setting up Python
# locally: data generation -> ring detection -> training -> decision layer ->
# adversarial hardening. `serving` runs the Day 7 /score API against model
# artifacts present in the mounted ./models and ./reports (run the pipeline
# target first, or mount pre-generated artifacts) — this is what docker-compose
# uses. `serving-standalone` is for a volume-less container host: it bundles the
# LLM extra and self-bootstraps the pipeline on first start. See DEPLOYMENT.md.

FROM python:3.11-slim AS base

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
RUN pip install --no-cache-dir -e .


FROM base AS pipeline

CMD ["sh", "-c", "python scripts/generate_data.py && python scripts/detect_rings.py && python scripts/train_baseline.py && python scripts/build_decision_layer.py && python scripts/run_adversarial_harness.py"]


FROM base AS serving

EXPOSE 8000
CMD ["uvicorn", "cerberus.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]


# `serving-standalone` — for a container host with NO mounted volumes (Render, Railway,
# Fly, a bare `docker run`). Adds the optional LLM extra and a self-bootstrapping
# entrypoint that runs the pipeline once if no model is baked in. See DEPLOYMENT.md.
#   docker build --target serving-standalone -t cerberus-serving-standalone .
FROM base AS serving-standalone

RUN pip install --no-cache-dir -e ".[llm]"
# Bake the model artifacts at build time: training runs on the (larger) build machine,
# so the running container only loads a ~1 MB booster + calibrator and stays well under
# a 512 MB free-tier RAM limit. The adversarial harness is skipped here — the API serves
# the baseline model, which is all /score and /explain need. serve.sh then sees the
# artifacts already present and goes straight to uvicorn.
RUN python scripts/generate_data.py \
 && python scripts/detect_rings.py \
 && python scripts/train_baseline.py \
 && python scripts/build_decision_layer.py
EXPOSE 8000
CMD ["sh", "scripts/serve.sh"]
