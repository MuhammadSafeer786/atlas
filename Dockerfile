# syntax=docker/dockerfile:1.7
# Multi-stage build: dependencies are built once in a builder image, the runtime
# image stays slim and runs as a non-root user.

FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY atlas ./atlas
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 ATLAS_DATA_DIR=/data
RUN useradd --create-home --uid 10001 atlas && mkdir -p /data && chown atlas /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=atlas examples ./examples
COPY --chown=atlas evals ./evals
USER atlas
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["atlas", "serve", "--host", "0.0.0.0", "--port", "8000"]
