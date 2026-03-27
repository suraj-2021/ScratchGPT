# =============================================================
#  DOCKERFILE — for Google Cloud Run
# =============================================================
#
# This file tells Docker how to package our entire application
# into a container that can be deployed anywhere.
#
# Build command:
#   docker build -t gpt2-chat .
#
# Run locally:
#   docker run -p 8080:8080 gpt2-chat
#
# Deploy to Cloud Run:
#   See deploy.sh or README.md for the full command.
# =============================================================

# ── Base image: Python 3.11 slim (smaller = faster to pull) ──
FROM python:3.11-slim

# ── Set working directory inside the container ────────────────
WORKDIR /app

# ── Environment variables ─────────────────────────────────────
# Prevents Python from writing .pyc files (not needed in containers)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs show up immediately)
ENV PYTHONUNBUFFERED=1
# Production mode
ENV DEBUG=False
# Cloud Run provides PORT env variable — default to 8080
ENV PORT=8080
# Tell HuggingFace to cache models inside /app (our working dir)
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# ── Install system dependencies ───────────────────────────────
# We need these for Python packages that have C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker layer caching trick) ─────
# If requirements.txt hasn't changed, Docker reuses this cached layer.
# This makes rebuilds much faster when you only change Python code.
COPY requirements.txt .

# ── Install Python packages ────────────────────────────────────
# --no-cache-dir: saves disk space
# torch CPU-only version is used here to keep image size manageable
# (Cloud Run free tier doesn't have GPUs anyway)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy the entire application code ─────────────────────────
COPY . .

# ── Run database migrations (creates db.sqlite3 for sessions) ─
RUN python manage.py migrate --run-syncdb

# ── Collect static files (CSS, JS) ────────────────────────────
RUN python manage.py collectstatic --noinput

# ── Download GPT-2 weights at build time ──────────────────────
# We download the weights during the Docker build so they're
# baked into the image. This means the container starts instantly
# without needing to download anything at runtime.
#
# NOTE: This makes the Docker image ~600MB larger.
#       Comment out this line if you prefer to download at first startup.
RUN python -c "from gpt.download import download_gpt2_weights; download_gpt2_weights()"

# ── Expose the port ────────────────────────────────────────────
EXPOSE 8080

# ── Start the app with Gunicorn ────────────────────────────────
# Gunicorn is a production-grade WSGI server.
# --workers 1: Cloud Run handles scaling via multiple container instances,
#              not multiple workers per container (model is huge in RAM).
# --threads 4: Handle multiple requests in parallel within one worker.
# --timeout 120: Allow up to 120s for the model to generate a response.
CMD gunicorn myproject.wsgi:application \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -