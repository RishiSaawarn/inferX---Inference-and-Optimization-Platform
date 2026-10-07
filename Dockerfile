FROM python:3.10-slim AS builder

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml .
# We'll copy source to avoid the package directory issue
COPY . .
# Install without deps first to cache them, or just install normal
RUN pip install --no-cache-dir .

# Download models during image build to bake them into the image
RUN python scripts/download_models.py

FROM python:3.10-slim

WORKDIR /app

# Non-root user
RUN useradd -m inferx_user
USER inferx_user

# Copy installed packages and models from builder
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /root/.cache/torch/hub/checkpoints/ /home/inferx_user/.cache/torch/hub/checkpoints/

COPY --chown=inferx_user:inferx_user . .

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
