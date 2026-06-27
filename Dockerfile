# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed for catboost / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# ─── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8080

# Train first if no model exists, then serve
CMD ["python", "app.py"]
