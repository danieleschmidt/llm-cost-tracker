FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "src.llm_cost_tracker.main:app", "--host", "0.0.0.0", "--port", "8000"]