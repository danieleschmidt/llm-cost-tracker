"""
Mock objects and utilities for testing LLM Cost Tracker.
"""

from unittest.mock import AsyncMock, MagicMock, Mock
from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, name: str, response_data: Dict[str, Any]):
        self.name = name
        self.response_data = response_data
        self.call_count = 0
        self.last_request = None
    
    async def create_completion(self, **kwargs) -> Dict[str, Any]:
        """Mock completion creation."""
        self.call_count += 1
        self.last_request = kwargs
        
        # Simulate API latency
        await asyncio.sleep(0.1)
        
        return self.response_data
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_request = None


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.query_count = 0
        self.last_query = None
    
    async def insert_cost_record(self, record: Dict[str, Any]) -> str:
        """Mock record insertion."""
        record_id = f"mock_id_{len(self.records)}"
        record["id"] = record_id
        record["created_at"] = datetime.now().isoformat()
        self.records.append(record)
        return record_id
    
    async def query_records(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Mock record querying."""
        self.query_count += 1
        self.last_query = filters
        
        if not filters:
            return self.records
        
        # Simple filtering logic for testing
        filtered = []
        for record in self.records:
            match = True
            for key, value in filters.items():
                if key in record and record[key] != value:
                    match = False
                    break
            if match:
                filtered.append(record)
        
        return filtered
    
    async def get_cost_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Mock cost summary."""
        total_cost = sum(record.get("cost_usd", 0) for record in self.records)
        total_tokens = sum(record.get("total_tokens", 0) for record in self.records)
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "request_count": len(self.records),
            "start_date": start_date,
            "end_date": end_date
        }
    
    def reset(self):
        """Reset mock database."""
        self.records = []
        self.query_count = 0
        self.last_query = None


class MockPrometheusRegistry:
    """Mock Prometheus registry for testing."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.collect_calls = 0
    
    def register(self, metric):
        """Mock metric registration."""
        self.metrics[metric.name] = metric
    
    def unregister(self, metric):
        """Mock metric unregistration."""
        if metric.name in self.metrics:
            del self.metrics[metric.name]
    
    def collect(self):
        """Mock metric collection."""
        self.collect_calls += 1
        return list(self.metrics.values())


class MockSlackWebhook:
    """Mock Slack webhook for testing."""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.call_count = 0
        self.should_fail = False
        self.failure_message = "Webhook failed"
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Mock message sending."""
        self.call_count += 1
        
        if self.should_fail:
            raise Exception(self.failure_message)
        
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
        
        return True
    
    def set_failure(self, should_fail: bool, message: str = "Webhook failed"):
        """Configure mock to simulate failures."""
        self.should_fail = should_fail
        self.failure_message = message
    
    def reset(self):
        """Reset mock state."""
        self.messages = []
        self.call_count = 0
        self.should_fail = False


class MockBudgetRulesEngine:
    """Mock budget rules engine for testing."""
    
    def __init__(self):
        self.current_spend = 0.0
        self.budget_limit = 1000.0
        self.alert_thresholds = [0.5, 0.8, 0.9]
        self.model_swap_rules = []
        self.triggered_alerts = []
        self.triggered_swaps = []
    
    def add_spend(self, amount: float):
        """Add spending amount."""
        self.current_spend += amount
        self._check_thresholds()
    
    def _check_thresholds(self):
        """Check alert thresholds."""
        current_ratio = self.current_spend / self.budget_limit
        
        for threshold in self.alert_thresholds:
            if current_ratio >= threshold and threshold not in [alert["threshold"] for alert in self.triggered_alerts]:
                self.triggered_alerts.append({
                    "threshold": threshold,
                    "current_spend": self.current_spend,
                    "budget_limit": self.budget_limit,
                    "timestamp": datetime.now().isoformat()
                })
    
    def should_swap_model(self, current_model: str) -> Optional[str]:
        """Check if model should be swapped."""
        current_ratio = self.current_spend / self.budget_limit
        
        for rule in self.model_swap_rules:
            if (rule["from_model"] == current_model and 
                current_ratio >= rule.get("threshold", 0.9)):
                
                self.triggered_swaps.append({
                    "from_model": current_model,
                    "to_model": rule["to_model"],
                    "threshold": rule.get("threshold", 0.9),
                    "current_ratio": current_ratio,
                    "timestamp": datetime.now().isoformat()
                })
                
                return rule["to_model"]
        
        return None
    
    def reset(self):
        """Reset budget engine state."""
        self.current_spend = 0.0
        self.triggered_alerts = []
        self.triggered_swaps = []


def create_mock_langchain_callback():
    """Create a mock LangChain callback."""
    callback = AsyncMock()
    
    # Mock all the callback methods
    callback.on_llm_start = AsyncMock()
    callback.on_llm_end = AsyncMock()
    callback.on_llm_error = AsyncMock()
    callback.on_chain_start = AsyncMock()
    callback.on_chain_end = AsyncMock()
    callback.on_chain_error = AsyncMock()
    callback.on_agent_action = AsyncMock()
    callback.on_agent_finish = AsyncMock()
    callback.on_tool_start = AsyncMock()
    callback.on_tool_end = AsyncMock()
    callback.on_tool_error = AsyncMock()
    
    return callback


def create_mock_opentelemetry_span():
    """Create a mock OpenTelemetry span."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.end = MagicMock()
    
    # Context manager support
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=None)
    
    return span


def create_mock_tracer():
    """Create a mock OpenTelemetry tracer."""
    tracer = MagicMock()
    span = create_mock_opentelemetry_span()
    tracer.start_span = MagicMock(return_value=span)
    
    return tracer


class MockMetricsCollector:
    """Mock metrics collector for testing."""
    
    def __init__(self):
        self.collected_metrics: List[Dict[str, Any]] = []
        self.collection_count = 0
    
    async def collect_metrics(self, metrics_data: Dict[str, Any]):
        """Mock metrics collection."""
        self.collection_count += 1
        self.collected_metrics.append({
            "timestamp": datetime.now().isoformat(),
            "data": metrics_data
        })
    
    def get_metrics_by_type(self, metric_type: str) -> List[Dict[str, Any]]:
        """Get metrics by type."""
        return [
            metric for metric in self.collected_metrics
            if metric["data"].get("type") == metric_type
        ]
    
    def reset(self):
        """Reset collector state."""
        self.collected_metrics = []
        self.collection_count = 0


# Convenience functions for creating common mocks
def create_mock_fastapi_request():
    """Create a mock FastAPI request object."""
    request = MagicMock()
    request.headers = {"user-agent": "test-client", "x-request-id": "test-123"}
    request.client.host = "127.0.0.1"
    request.method = "POST"
    request.url.path = "/api/v1/track"
    
    return request


def create_mock_async_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.scalar = AsyncMock()
    
    return session