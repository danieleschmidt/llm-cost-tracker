"""Tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from llm_cost_tracker.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM Cost Tracker API" in response.json()["message"]