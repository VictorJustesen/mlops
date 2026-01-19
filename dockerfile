FROM python:3.13-slim AS base
WORKDIR /app

# Install uv
RUN pip install uv

# Copy lockfile and sync dependencies
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-editable

COPY src/ src/
ENV PYTHONPATH=/app

# Trainer stage - for training models
FROM base AS trainer
COPY data/ data/
# Create models directory
RUN mkdir -p models
# CMD will be overridden or use default training
CMD ["uv", "run", "python", "-u", "src/models/train_rnn.py"]

# API stage - for serving predictions
FROM base AS api
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# Create models directory (will be populated from GCS or local mount)
RUN mkdir -p models
ENV PORT=8080
ENV GCS_MODEL_BUCKET="mlops-dataset-84636"
ENV GCS_MODEL_PATH="models"
# MODEL_SOURCE: "gcs" (download from GCS) or "local" (use mounted models/)
ENV MODEL_SOURCE="gcs"
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
