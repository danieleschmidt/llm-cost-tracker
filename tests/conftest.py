"""
Pytest configuration and shared fixtures for LLM Cost Tracker tests.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from llm_cost_tracker.config import Settings
from llm_cost_tracker.database import Base, get_db
from llm_cost_tracker.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with in-memory database."""
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        log_level="DEBUG",
        secret_key="test_secret_key_do_not_use_in_production",
        enable_budget_alerts=True,
        enable_model_swapping=False,  # Disable for tests
        enable_metrics_export=False,  # Disable for tests
    )


@pytest.fixture
async def test_db_engine(test_settings: Settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.database_url,
        echo=test_settings.debug,
        future=True,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def test_client(test_settings: Settings, test_db_session: AsyncSession) -> TestClient:
    """Create test client with overridden dependencies."""
    
    def override_get_db():
        return test_db_session
    
    def override_get_settings():
        return test_settings
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[Settings] = override_get_settings
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Test response"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
async def mock_langchain_callback():
    """Mock LangChain callback for testing."""
    callback = AsyncMock()
    callback.on_llm_start = AsyncMock()
    callback.on_llm_end = AsyncMock()
    callback.on_llm_error = AsyncMock()
    return callback


@pytest.fixture
def sample_cost_data():
    """Sample cost data for testing."""
    return {
        "request_id": "test-request-123",
        "timestamp": "2024-01-15T10:30:00Z",
        "model": "gpt-4",
        "provider": "openai",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost_usd": 0.0045,
        "latency_ms": 1250,
        "user_id": "test-user",
        "application": "test-app",
        "tags": {"environment": "test", "version": "1.0.0"},
    }


@pytest.fixture
def sample_budget_config():
    """Sample budget configuration for testing."""
    return {
        "monthly_budget": 1000.0,
        "alert_thresholds": [0.5, 0.8, 0.9],
        "model_swap_rules": [
            {
                "condition": "cost_threshold > 0.9",
                "from_model": "gpt-4",
                "to_model": "gpt-3.5-turbo",
            }
        ],
        "notification_channels": {
            "slack": {
                "webhook_url": "https://hooks.slack.com/test",
                "enabled": True,
            }
        },
    }


@pytest.fixture
def temporary_config_file():
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("""
monthly_budget: 500.0
alert_thresholds:
  - 0.7
  - 0.85
  - 0.95
model_swap_rules:
  - condition: "cost_threshold > 0.8"
    from_model: "gpt-4"
    to_model: "gpt-3.5-turbo"
notification_channels:
  slack:
    webhook_url: "https://hooks.slack.com/test"
    enabled: true
""")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry for testing."""
    from unittest.mock import patch
    
    with patch("prometheus_client.CollectorRegistry") as mock_registry:
        mock_registry.return_value = MagicMock()
        yield mock_registry


@pytest.fixture
def mock_otel_tracer():
    """Mock OpenTelemetry tracer for testing."""
    from unittest.mock import patch
    
    with patch("opentelemetry.trace.get_tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.return_value.start_span.return_value.__enter__.return_value = mock_span
        yield mock_tracer


@pytest.fixture(autouse=True)
def isolate_tests():
    """Ensure test isolation by resetting global state."""
    # Reset any global variables or singletons
    yield
    # Cleanup after test


# Performance testing fixtures
@pytest.fixture
def performance_data_generator():
    """Generate performance test data."""
    def generate(count: int = 1000):
        import random
        from datetime import datetime, timedelta
        
        data = []
        base_time = datetime.now()
        
        for i in range(count):
            data.append({
                "request_id": f"perf-test-{i}",
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3"]),
                "prompt_tokens": random.randint(10, 1000),
                "completion_tokens": random.randint(5, 500),
                "cost_usd": round(random.uniform(0.001, 0.1), 6),
                "latency_ms": random.randint(100, 5000),
            })
        
        return data
    
    return generate


# Integration test fixtures
@pytest.fixture
async def integration_test_setup():
    """Setup for integration tests."""
    # Start test services (database, Redis, etc.)
    yield
    # Cleanup test services


# E2E test fixtures
@pytest.fixture(scope="session")
def docker_compose_file():
    """Path to docker-compose file for E2E tests."""
    return os.path.join(os.path.dirname(__file__), "..", "docker-compose.test.yml")


@pytest.fixture(scope="session")
def docker_services():
    """Start Docker services for E2E tests."""
    # This would integrate with pytest-docker if needed
    pass


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "security: Security-related tests")


# Test data cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Cleanup test data after each test."""
    yield
    # Perform cleanup operations here
    pass