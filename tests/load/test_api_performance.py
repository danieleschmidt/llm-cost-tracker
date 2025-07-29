"""
Load testing configuration for LLM Cost Tracker API endpoints.

This module provides comprehensive load testing scenarios using Locust
to validate system performance under various load conditions.
"""

import json
import random
import time
from typing import Dict, Any

from locust import HttpUser, task, between
from faker import Faker

fake = Faker()


class LLMCostTrackerUser(HttpUser):
    """
    Simulates a user interacting with the LLM Cost Tracker API.
    
    This class defines realistic user behavior patterns for load testing
    the cost tracking endpoints, metrics collection, and dashboard queries.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize test data when the user starts."""
        self.application_name = f"test-app-{fake.uuid4()[:8]}"
        self.user_id = f"user-{fake.uuid4()[:8]}"
        self.models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-haiku"]
        self.test_data = self._generate_test_traces()
    
    def _generate_test_traces(self) -> list[Dict[str, Any]]:
        """Generate realistic trace data for testing."""
        traces = []
        for _ in range(10):
            trace = {
                "trace_id": fake.uuid4(),
                "span_id": fake.uuid4(),
                "application": self.application_name,
                "user_id": self.user_id,
                "model": random.choice(self.models),
                "prompt_tokens": random.randint(50, 500),
                "completion_tokens": random.randint(20, 200),
                "total_tokens": 0,  # Will be calculated
                "latency_ms": random.randint(200, 2000),
                "cost_usd": round(random.uniform(0.001, 0.05), 6),
                "timestamp": int(time.time() * 1000),
                "metadata": {
                    "temperature": round(random.uniform(0.1, 1.0), 2),
                    "max_tokens": random.randint(100, 1000),
                    "provider": random.choice(["openai", "anthropic", "cohere"])
                }
            }
            trace["total_tokens"] = trace["prompt_tokens"] + trace["completion_tokens"]
            traces.append(trace)
        return traces
    
    @task(3)
    def submit_trace(self):
        """Submit trace data to the metrics endpoint."""
        trace_data = random.choice(self.test_data)
        
        with self.client.post(
            "/v1/traces",
            json=trace_data,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Failed to submit trace: {response.status_code}")
    
    @task(2)
    def get_metrics_summary(self):
        """Fetch metrics summary for the application."""
        params = {
            "application": self.application_name,
            "time_range": "1h",
            "group_by": "model"
        }
        
        with self.client.get(
            "/v1/metrics/summary",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get metrics: {response.status_code}")
    
    @task(2)
    def get_cost_breakdown(self):
        """Fetch cost breakdown by model and time period."""
        params = {
            "application": self.application_name,
            "start_time": int((time.time() - 3600) * 1000),  # Last hour
            "end_time": int(time.time() * 1000),
            "group_by": "model,user_id"
        }
        
        with self.client.get(
            "/v1/costs/breakdown",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "breakdown" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Failed to get cost breakdown: {response.status_code}")
    
    @task(1)
    def get_budget_status(self):
        """Check budget status and alerts."""
        with self.client.get(
            f"/v1/budget/status/{self.application_name}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get budget status: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Perform health check to ensure system availability."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_prometheus_metrics(self):
        """Fetch Prometheus metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get Prometheus metrics: {response.status_code}")


class HighVolumeUser(HttpUser):
    """
    Simulates high-volume batch processing scenarios.
    
    This user type sends larger batches of trace data to test
    system behavior under sustained high load.
    """
    
    wait_time = between(0.1, 1.0)  # More aggressive timing
    
    def on_start(self):
        """Initialize high-volume test scenario."""
        self.application_name = f"batch-app-{fake.uuid4()[:8]}"
        self.batch_size = random.randint(10, 50)
    
    @task
    def submit_batch_traces(self):
        """Submit a batch of traces simultaneously."""
        traces = []
        for _ in range(self.batch_size):
            trace = {
                "trace_id": fake.uuid4(),
                "span_id": fake.uuid4(),
                "application": self.application_name,
                "user_id": f"batch-user-{fake.uuid4()[:8]}",
                "model": random.choice(["gpt-3.5-turbo", "gpt-4"]),
                "prompt_tokens": random.randint(100, 1000),
                "completion_tokens": random.randint(50, 500),
                "latency_ms": random.randint(300, 1500),
                "cost_usd": round(random.uniform(0.005, 0.1), 6),
                "timestamp": int(time.time() * 1000)
            }
            trace["total_tokens"] = trace["prompt_tokens"] + trace["completion_tokens"]
            traces.append(trace)
        
        with self.client.post(
            "/v1/traces/batch",
            json={"traces": traces},
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code in [201, 202]:
                response.success()
            else:
                response.failure(f"Batch submission failed: {response.status_code}")


class DashboardUser(HttpUser):
    """
    Simulates dashboard and reporting queries.
    
    This user type focuses on read-heavy operations that would
    be typical of users viewing dashboards and generating reports.
    """
    
    wait_time = between(2, 8)  # Dashboard users browse more slowly
    
    def on_start(self):
        """Initialize dashboard user session."""
        self.applications = [f"app-{i}" for i in range(1, 6)]
        self.time_ranges = ["1h", "6h", "1d", "7d", "30d"]
    
    @task(4)
    def dashboard_overview(self):
        """Load main dashboard overview."""
        params = {
            "time_range": random.choice(self.time_ranges),
            "applications": ",".join(random.sample(self.applications, 2))
        }
        
        with self.client.get(
            "/v1/dashboard/overview",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard overview failed: {response.status_code}")
    
    @task(2)
    def cost_trends(self):
        """Fetch cost trend data for charts."""
        params = {
            "application": random.choice(self.applications),
            "time_range": random.choice(self.time_ranges),
            "interval": "1h"
        }
        
        with self.client.get(
            "/v1/analytics/trends",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Cost trends failed: {response.status_code}")
    
    @task(1)
    def export_report(self):
        """Generate and download cost report."""
        params = {
            "format": random.choice(["csv", "json", "xlsx"]),
            "time_range": "7d",
            "include_details": "true"
        }
        
        with self.client.get(
            "/v1/reports/export",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Report export failed: {response.status_code}")