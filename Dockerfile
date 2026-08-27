# Reproducible pipeline container: generate synthetic data, train the baseline
# point-risk model, run the ring detector. A reviewer with only Docker installed
# can reproduce every number in the README without setting up Python locally.
#
# Serving (FastAPI) gets its own stage once the Day 7 API exists — this image
# is the offline pipeline, not the request-serving path.

FROM python:3.11-slim AS pipeline

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
RUN pip install --no-cache-dir -e .

CMD ["sh", "-c", "python scripts/generate_data.py && python scripts/detect_rings.py && python scripts/train_baseline.py"]
