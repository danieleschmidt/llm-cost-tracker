"""
Sample data fixtures for testing LLM Cost Tracker functionality.
"""

from datetime import datetime, timedelta
from typing import Dict, List
import json


# Sample LLM provider responses
SAMPLE_OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I assist you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 10,
        "total_tokens": 25
    }
}

SAMPLE_ANTHROPIC_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'm Claude, an AI assistant."
        }
    ],
    "model": "claude-3-sonnet-20240229",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 8
    }
}

# Cost tracking data samples
SAMPLE_COST_RECORDS = [
    {
        "request_id": "req_001",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt-4",
        "provider": "openai",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost_usd": 0.0045,
        "latency_ms": 1250,
        "user_id": "user_1",
        "application": "chat_app",
        "tags": {"environment": "test"}
    },
    {
        "request_id": "req_002",
        "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
        "model": "claude-3-sonnet",
        "provider": "anthropic",
        "prompt_tokens": 80,
        "completion_tokens": 40,
        "total_tokens": 120,
        "cost_usd": 0.0036,
        "latency_ms": 950,
        "user_id": "user_2",
        "application": "summarizer",
        "tags": {"environment": "test", "version": "1.2.0"}
    },
    {
        "request_id": "req_003",
        "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
        "model": "gpt-3.5-turbo",
        "provider": "openai",
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
        "cost_usd": 0.0009,
        "latency_ms": 800,
        "user_id": "user_1",
        "application": "chat_app",
        "tags": {"environment": "test"}
    }
]

# Budget configuration samples
SAMPLE_BUDGET_CONFIGS = {
    "basic": {
        "monthly_budget": 500.0,
        "alert_thresholds": [0.5, 0.8, 0.9],
        "model_swap_rules": [],
        "notification_channels": {
            "email": {"enabled": True, "recipients": ["admin@test.com"]}
        }
    },
    "advanced": {
        "monthly_budget": 2000.0,
        "alert_thresholds": [0.6, 0.8, 0.95],
        "model_swap_rules": [
            {
                "condition": "cost_threshold > 0.8",
                "from_model": "gpt-4",
                "to_model": "gpt-3.5-turbo",
                "priority": 1
            },
            {
                "condition": "cost_threshold > 0.9",
                "from_model": "claude-3-opus",
                "to_model": "claude-3-sonnet",
                "priority": 2
            }
        ],
        "notification_channels": {
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/test"
            },
            "email": {
                "enabled": True,
                "recipients": ["admin@test.com", "devops@test.com"]
            }
        }
    }
}

# Performance test data
def generate_performance_data(count: int = 1000) -> List[Dict]:
    """Generate performance test data with realistic distributions."""
    import random
    
    models = [
        ("gpt-4", 0.03, 0.06, 1500, 3000),  # (model, input_cost, output_cost, min_latency, max_latency)
        ("gpt-3.5-turbo", 0.001, 0.002, 800, 2000),
        ("claude-3-sonnet", 0.003, 0.015, 1000, 2500),
        ("claude-3-haiku", 0.00025, 0.00125, 500, 1500)
    ]
    
    applications = ["chat_app", "summarizer", "translator", "analyzer", "generator"]
    users = [f"user_{i}" for i in range(1, 21)]  # 20 users
    
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(count):
        model, input_cost, output_cost, min_lat, max_lat = random.choice(models)
        
        prompt_tokens = random.randint(10, 2000)
        completion_tokens = random.randint(5, 1000)
        total_tokens = prompt_tokens + completion_tokens
        
        cost = (prompt_tokens * input_cost / 1000) + (completion_tokens * output_cost / 1000)
        latency = random.randint(min_lat, max_lat)
        
        timestamp = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        data.append({
            "request_id": f"perf_test_{i:06d}",
            "timestamp": timestamp.isoformat(),
            "model": model,
            "provider": model.split("-")[0] if "-" in model else "openai",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": latency,
            "user_id": random.choice(users),
            "application": random.choice(applications),
            "tags": {
                "environment": "performance_test",
                "batch": f"batch_{i // 100}",
                "priority": random.choice(["low", "medium", "high"])
            }
        })
    
    return data

# Error scenarios for testing
ERROR_SCENARIOS = {
    "network_timeout": {
        "error_type": "TimeoutError",
        "message": "Request timed out after 30 seconds",
        "recoverable": True
    },
    "rate_limit": {
        "error_type": "RateLimitError", 
        "message": "Rate limit exceeded. Please try again later.",
        "recoverable": True,
        "retry_after": 60
    },
    "invalid_api_key": {
        "error_type": "AuthenticationError",
        "message": "Invalid API key provided",
        "recoverable": False
    },
    "quota_exceeded": {
        "error_type": "QuotaExceededError",
        "message": "Monthly quota exceeded",
        "recoverable": False
    },
    "model_unavailable": {
        "error_type": "ModelUnavailableError",
        "message": "The requested model is temporarily unavailable",
        "recoverable": True
    }
}

# Model pricing data for testing
MODEL_PRICING = {
    "gpt-4": {
        "provider": "openai",
        "input_cost_per_1k": 0.03,
        "output_cost_per_1k": 0.06,
        "context_window": 8192,
        "max_output_tokens": 4096
    },
    "gpt-3.5-turbo": {
        "provider": "openai", 
        "input_cost_per_1k": 0.001,
        "output_cost_per_1k": 0.002,
        "context_window": 16385,
        "max_output_tokens": 4096
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "input_cost_per_1k": 0.015,
        "output_cost_per_1k": 0.075,
        "context_window": 200000,
        "max_output_tokens": 4096
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015,
        "context_window": 200000,
        "max_output_tokens": 4096
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125,
        "context_window": 200000,
        "max_output_tokens": 4096
    }
}

# Configuration templates for different environments
ENVIRONMENT_CONFIGS = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite+aiosqlite:///./test.db",
        "enable_metrics_export": True,
        "enable_budget_alerts": False,
        "enable_model_swapping": False
    },
    "testing": {
        "debug": True,
        "log_level": "INFO",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "enable_metrics_export": False,
        "enable_budget_alerts": True,
        "enable_model_swapping": True
    },
    "staging": {
        "debug": False,
        "log_level": "INFO",
        "database_url": "postgresql+asyncpg://user:pass@localhost/llm_metrics_staging",
        "enable_metrics_export": True,
        "enable_budget_alerts": True,
        "enable_model_swapping": True
    },
    "production": {
        "debug": False,
        "log_level": "WARNING",
        "database_url": "postgresql+asyncpg://user:pass@localhost/llm_metrics",
        "enable_metrics_export": True,
        "enable_budget_alerts": True,
        "enable_model_swapping": True
    }
}