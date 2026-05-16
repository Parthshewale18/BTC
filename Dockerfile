# =============================================================================
# Dockerfile — Bitcoin ML Prediction API
# =============================================================================
#
# What this file does:
# --------------------
# Packages the entire project into a Docker container so it runs
# identically on any machine — your laptop, a server, or Render.
#
# Think of Docker like a shipping container:
# "It worked on my machine" problem disappears because the container
# carries its own OS, Python version, packages, and code.
#
# Build the image:
#   docker build -t bitcoin-ml .
#
# Run locally:
#   docker run -p 8000:8000 bitcoin-ml
#
# Then open: http://localhost:8000/docs
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1 — Base image
# -----------------------------------------------------------------------------
# We use Python 3.11 slim (not full) to keep image size small.
# "slim" has just enough OS to run Python — no compilers, no extras.
# This matters because large images are slow to deploy on Render.
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# STAGE 2 — Set environment variables
# -----------------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE=1  → don't create .pyc files (saves space)
# PYTHONUNBUFFERED=1         → print logs immediately (don't buffer)
#                              critical for seeing logs in Render dashboard
# PYTHONIOENCODING=utf-8     → fixes Windows-style Unicode errors
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

# -----------------------------------------------------------------------------
# STAGE 3 — Install system dependencies
# -----------------------------------------------------------------------------
# Some Python packages (like numpy, scikit-learn) need C libraries.
# We install them here before installing Python packages.
# --no-install-recommends keeps the image lean.
# We clean up apt cache at the end to reduce image size.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# libgomp1 is required by XGBoost for parallel tree building (OpenMP)

# -----------------------------------------------------------------------------
# STAGE 4 — Set working directory
# -----------------------------------------------------------------------------
# All subsequent commands run from /app inside the container.
# This is the root of our project inside the container.
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# STAGE 5 — Install Python dependencies
# -----------------------------------------------------------------------------
# We copy requirements.txt FIRST (before copying our code).
# Why? Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips reinstalling packages and uses the cached layer.
# This makes rebuilds much faster during development.
# -----------------------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --no-cache-dir → don't store pip's download cache in the image (saves space)

# -----------------------------------------------------------------------------
# STAGE 6 — Copy project files into container
# -----------------------------------------------------------------------------
# We copy specific folders (not everything) to keep the image clean.
# .dockerignore (below) also excludes unnecessary files.
# -----------------------------------------------------------------------------
COPY src/      ./src/
COPY app/      ./app/
COPY models/   ./models/

# Create directories that need to exist at runtime
RUN mkdir -p logs data/raw data/processed

# -----------------------------------------------------------------------------
# STAGE 7 — Expose port
# -----------------------------------------------------------------------------
# Tell Docker this container listens on port 8000.
# This does NOT actually publish the port — that's done with -p at runtime.
# Render reads this to know which port to forward traffic to.
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# STAGE 8 — Health check
# -----------------------------------------------------------------------------
# Docker (and Render) will call this every 30 seconds.
# If it fails 3 times in a row, Docker considers the container unhealthy.
# We call our /health endpoint — if the API is running, it returns 200.
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------------------------------
# STAGE 9 — Start command
# -----------------------------------------------------------------------------
# This runs when the container starts.
# uvicorn is the ASGI server that runs our FastAPI app.
#
# --host 0.0.0.0  → listen on all network interfaces (required for Docker)
#                   without this, the app only accepts connections from inside
#                   the container and is unreachable from outside
# --port 8000     → port to listen on
# --workers 2     → 2 parallel worker processes (handles concurrent requests)
# --log-level info→ show info logs in Render dashboard
# -----------------------------------------------------------------------------
CMD ["uvicorn", "app.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]