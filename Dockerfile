# Two build targets from one image:
#   docker build --target pipeline -t cerberus-pipeline .   (default)
#   docker build --target serving  -t cerberus-serving .
#
# `pipeline` reproduces every number in the README without setting up Python
# locally: data generation -> ring detection -> training -> decision layer ->
# adversarial hardening. `serving` runs the Day 7 /score API against whatever
# model artifacts are present in the mounted ./models and ./reports (run the
# pipeline target first, or mount pre-generated artifacts).

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
