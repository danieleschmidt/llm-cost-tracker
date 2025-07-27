"""
End-to-end tests for complete LLM Cost Tracker workflows.
"""

import pytest
import asyncio
import json
from typing import Dict, Any


class TestFullWorkflow:
    """End-to-end tests for complete system workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_cost_tracking_flow(self, test_client, sample_cost_data):
        """Test complete flow from cost data ingestion to alerting."""
        
        # Step 1: Ingest cost data
        response = test_client.post("/api/v1/costs", json=sample_cost_data)
        assert response.status_code == 201
        ingestion_result = response.json()
        assert "request_id" in ingestion_result
        
        # Step 2: Verify data is stored
        response = test_client.get(f"/api/v1/costs/{ingestion_result['request_id']}")
        assert response.status_code == 200
        stored_data = response.json()
        assert stored_data["model"] == sample_cost_data["model"]
        assert stored_data["cost_usd"] == sample_cost_data["cost_usd"]
        
        # Step 3: Check metrics endpoint
        response = test_client.get("/metrics")
        assert response.status_code == 200
        metrics_data = response.text
        assert "llm_cost_total" in metrics_data
        
        # Step 4: Verify budget tracking
        response = test_client.get("/api/v1/budget/status")
        assert response.status_code == 200
        budget_status = response.json()
        assert "current_spend" in budget_status
        assert "remaining_budget" in budget_status

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_budget_alert_workflow(self, test_client, sample_budget_config):
        """Test budget alert generation and notification workflow."""
        
        # Step 1: Configure budget
        response = test_client.put("/api/v1/budget/config", json=sample_budget_config)
        assert response.status_code == 200
        
        # Step 2: Simulate high cost usage to trigger alert
        high_cost_data = {
            "request_id": "high-cost-request",
            "model": "gpt-4",
            "cost_usd": 800.0,  # High cost to trigger alert
            "timestamp": "2024-01-15T10:30:00Z",
            "prompt_tokens": 10000,
            "completion_tokens": 5000,
            "total_tokens": 15000,
            "provider": "openai",
        }
        
        response = test_client.post("/api/v1/costs", json=high_cost_data)
        assert response.status_code == 201
        
        # Step 3: Check if alert was triggered
        response = test_client.get("/api/v1/alerts")
        assert response.status_code == 200
        alerts = response.json()
        assert len(alerts) > 0
        assert any(alert["type"] == "budget_threshold" for alert in alerts)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_model_swapping_workflow(self, test_client, sample_budget_config):
        """Test automatic model swapping when cost thresholds are exceeded."""
        
        # Step 1: Configure budget with model swap rules
        config_with_swap = {
            **sample_budget_config,
            "model_swap_rules": [
                {
                    "condition": "cost_threshold > 0.5",
                    "from_model": "gpt-4",
                    "to_model": "gpt-3.5-turbo",
                }
            ]
        }
        
        response = test_client.put("/api/v1/budget/config", json=config_with_swap)
        assert response.status_code == 200
        
        # Step 2: Simulate costs that trigger model swap
        for i in range(10):
            cost_data = {
                "request_id": f"swap-trigger-{i}",
                "model": "gpt-4",
                "cost_usd": 60.0,  # Will accumulate to trigger swap
                "timestamp": f"2024-01-15T10:{30+i}:00Z",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "provider": "openai",
            }
            
            response = test_client.post("/api/v1/costs", json=cost_data)
            assert response.status_code == 201
        
        # Step 3: Check model recommendations
        response = test_client.get("/api/v1/models/recommendations")
        assert response.status_code == 200
        recommendations = response.json()
        
        # Should recommend cheaper model due to cost threshold
        assert any(
            rec["recommended_model"] == "gpt-3.5-turbo" 
            for rec in recommendations
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_monitoring_and_observability_flow(self, test_client):
        """Test monitoring, metrics, and observability features."""
        
        # Step 1: Health check
        response = test_client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Readiness check
        response = test_client.get("/ready")
        assert response.status_code == 200
        
        # Step 3: Prometheus metrics
        response = test_client.get("/metrics")
        assert response.status_code == 200
        metrics = response.text
        
        # Verify key metrics are present
        expected_metrics = [
            "llm_requests_total",
            "llm_cost_total",
            "llm_latency_seconds",
            "llm_tokens_total",
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Step 4: OpenTelemetry traces (if available)
        # This would verify trace data is being generated

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_api_authentication_and_authorization(self, test_client):
        """Test API authentication and authorization workflows."""
        
        # Step 1: Test unauthenticated request (if auth is enabled)
        response = test_client.get("/api/v1/costs")
        # Depending on configuration, this might return 401 or 200
        assert response.status_code in [200, 401]
        
        # Step 2: Test with valid API key (if configured)
        headers = {"X-API-Key": "test-api-key"}
        response = test_client.get("/api/v1/costs", headers=headers)
        # Should work with valid key
        assert response.status_code in [200, 401]  # Depends on test configuration

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_data_export_workflow(self, test_client, sample_cost_data):
        """Test data export functionality."""
        
        # Step 1: Ingest some test data
        for i in range(5):
            data = {
                **sample_cost_data,
                "request_id": f"export-test-{i}",
                "timestamp": f"2024-01-15T10:{30+i}:00Z",
            }
            response = test_client.post("/api/v1/costs", json=data)
            assert response.status_code == 201
        
        # Step 2: Export data as CSV
        response = test_client.get("/api/v1/export/csv?start_date=2024-01-15&end_date=2024-01-16")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        
        # Step 3: Export data as JSON
        response = test_client.get("/api/v1/export/json?start_date=2024-01-15&end_date=2024-01-16")
        assert response.status_code == 200
        export_data = response.json()
        assert len(export_data) == 5

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_system_under_sustained_load(self, test_client, performance_data_generator):
        """Test system behavior under sustained load."""
        
        # Generate sustained load
        test_data = performance_data_generator(100)
        
        # Send requests over time to simulate sustained load
        for batch_start in range(0, len(test_data), 10):
            batch = test_data[batch_start:batch_start + 10]
            
            # Send batch of requests
            for item in batch:
                response = test_client.post("/api/v1/costs", json=item)
                assert response.status_code == 201
            
            # Small delay to simulate realistic load
            await asyncio.sleep(0.1)
        
        # Verify system is still responsive
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # Verify all data was processed
        response = test_client.get("/api/v1/costs/count")
        assert response.status_code == 200
        count_data = response.json()
        assert count_data["count"] >= len(test_data)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_client):
        """Test system error handling and recovery mechanisms."""
        
        # Step 1: Test invalid data handling
        invalid_data = {"invalid": "data", "missing": "required_fields"}
        response = test_client.post("/api/v1/costs", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        # Step 2: Test system recovery after error
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # Step 3: Test rate limiting (if implemented)
        # Make many requests quickly to test rate limiting
        for _ in range(20):
            response = test_client.get("/api/v1/costs")
            # Should eventually get rate limited or continue working
            assert response.status_code in [200, 429]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_configuration_management(self, test_client, temporary_config_file):
        """Test configuration management and hot reloading."""
        
        # Step 1: Update configuration
        new_config = {
            "monthly_budget": 2000.0,
            "alert_thresholds": [0.6, 0.8, 0.95],
        }
        
        response = test_client.put("/api/v1/config", json=new_config)
        assert response.status_code == 200
        
        # Step 2: Verify configuration was updated
        response = test_client.get("/api/v1/config")
        assert response.status_code == 200
        config_data = response.json()
        assert config_data["monthly_budget"] == 2000.0