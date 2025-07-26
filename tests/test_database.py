"""Tests for database operations."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_cost_tracker.database import DatabaseManager


@pytest.fixture
def mock_pool():
    """Mock database pool."""
    pool = AsyncMock()
    connection = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = connection
    pool.acquire.return_value.__aexit__ = AsyncMock()
    return pool, connection


@pytest.mark.asyncio
async def test_store_span(mock_pool):
    """Test storing span data."""
    pool, connection = mock_pool
    connection.fetchval.return_value = "test_span_id"
    
    db = DatabaseManager()
    db.pool = pool
    
    span_data = {
        "span_id": "test_span",
        "trace_id": "test_trace",
        "operation_name": "test_op",
        "start_time": datetime.now(),
        "duration_ms": 100,
        "attributes": {"test": "data"}
    }
    
    result = await db.store_span(span_data)
    
    assert result == "test_span_id"
    connection.fetchval.assert_called_once()


@pytest.mark.asyncio
async def test_store_llm_metrics(mock_pool):
    """Test storing LLM metrics."""
    pool, connection = mock_pool
    connection.fetchval.return_value = "metric_id"
    
    db = DatabaseManager()
    db.pool = pool
    
    metrics_data = {
        "model_name": "gpt-4",
        "provider": "openai",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_cost_usd": 0.002,
        "application_name": "test_app"
    }
    
    result = await db.store_llm_metrics(metrics_data)
    
    assert result == "metric_id"
    connection.fetchval.assert_called_once()


@pytest.mark.asyncio
async def test_get_metrics_summary(mock_pool):
    """Test getting metrics summary."""
    pool, connection = mock_pool
    connection.fetchrow.return_value = {
        "total_requests": 100,
        "total_cost": 1.50,
        "avg_latency": 500.0,
        "unique_models": 3,
        "unique_applications": 2
    }
    
    db = DatabaseManager()
    db.pool = pool
    
    result = await db.get_metrics_summary(hours=24)
    
    expected = {
        "total_requests": 100,
        "total_cost": 1.50,
        "avg_latency": 500.0,
        "unique_models": 3,
        "unique_applications": 2,
        "period_hours": 24
    }
    
    assert result == expected
    connection.fetchrow.assert_called_once()