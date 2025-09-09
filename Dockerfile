# SecurePayQR Docker Configuration
# Multi-stage build for production optimization

# Stage 1: Base Python image with ML dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Training environment
FROM base as training

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories for data and outputs
RUN mkdir -p data models outputs logs

# Default command for training
CMD ["python", "src/training_pipeline.py", "--config", "config/train_config.json"]

# Stage 3: API production environment
FROM base as production

WORKDIR /app

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY src/fastapi_backend.py ./main.py
COPY models/ ./models/
COPY static/ ./static/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]

# Stage 4: Development environment
FROM base as development

WORKDIR /app

# Install all dependencies including dev tools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install development tools
RUN pip install \
    jupyter \
    black \
    flake8 \
    pytest \
    pytest-asyncio \
    ipython

# Copy all source code
COPY . .

# Expose ports for API and Jupyter
EXPOSE 8000 8888

# Development command
CMD ["uvicorn", "src.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]