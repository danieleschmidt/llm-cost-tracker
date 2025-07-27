# Multi-stage Dockerfile for LLM Cost Tracker
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Stage 2: Development environment
FROM builder as development

# Install development dependencies
RUN poetry install --no-interaction --no-ansi

# Copy source code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Development command with hot reload
CMD ["uvicorn", "llm_cost_tracker.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production runtime
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /bin/bash appuser \
    && mkdir -p /app /home/appuser/.local/bin \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser pyproject.toml ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "llm_cost_tracker.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 4: Security scanning
FROM production as security

USER root

# Install security scanning tools
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for vulnerability scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - \
    && echo "deb https://aquasecurity.github.io/trivy-repo/deb generic main" | tee -a /etc/apt/sources.list.d/trivy.list \
    && apt-get update && apt-get install -y trivy

# Run security scan (this will be used in CI)
RUN trivy fs --exit-code 1 --no-progress --security-checks vuln /app || true

USER appuser