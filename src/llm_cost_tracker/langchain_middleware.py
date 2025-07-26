"""OpenTelemetry middleware for LangChain callbacks."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import get_settings
from .database import db_manager

logger = logging.getLogger(__name__)


class LLMCostTracker(AsyncCallbackHandler):
    """Async callback handler for tracking LLM usage and costs."""
    
    def __init__(self, application_name: str = "default", user_id: Optional[str] = None):
        super().__init__()
        self.application_name = application_name
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
        self.active_runs: Dict[str, Dict] = {}
        
        # Initialize OpenTelemetry
        self._setup_otel()
        
        # Model cost mapping (per 1K tokens)
        self.model_costs = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-1.5-pro": {"input": 0.007, "output": 0.021},
        }
    
    def _setup_otel(self) -> None:
        """Setup OpenTelemetry tracing."""
        resource = Resource(attributes={
            "service.name": get_settings().otel_service_name,
            "service.version": "0.1.0",
            "application.name": self.application_name
        })
        
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=get_settings().otel_endpoint,
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
    
    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """Extract model name from serialized LLM data."""
        if "kwargs" in serialized:
            model = serialized["kwargs"].get("model_name") or serialized["kwargs"].get("model")
            if model:
                return model
        
        # Fallback to class name
        return serialized.get("id", ["", "", "unknown"])[-1]
    
    def _calculate_costs(self, model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate costs based on token usage."""
        costs = self.model_costs.get(model_name.lower(), {"input": 0.001, "output": 0.002})
        
        prompt_cost = (input_tokens / 1000) * costs["input"]
        completion_cost = (output_tokens / 1000) * costs["output"]
        
        return {
            "prompt_cost_usd": round(prompt_cost, 6),
            "completion_cost_usd": round(completion_cost, 6),
            "total_cost_usd": round(prompt_cost + completion_cost, 6)
        }
    
    def _extract_token_usage(self, llm_result: LLMResult) -> Dict[str, int]:
        """Extract token usage from LLM result."""
        if not llm_result.llm_output:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        usage = llm_result.llm_output.get("token_usage", {})
        
        # Handle different token usage formats
        input_tokens = (
            usage.get("prompt_tokens") or 
            usage.get("input_tokens") or 
            usage.get("tokens_prompt") or 0
        )
        
        output_tokens = (
            usage.get("completion_tokens") or 
            usage.get("output_tokens") or 
            usage.get("tokens_completion") or 0
        )
        
        total_tokens = (
            usage.get("total_tokens") or 
            input_tokens + output_tokens
        )
        
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens)
        }
    
    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts running."""
        run_id = kwargs.get("run_id")
        if not run_id:
            return
        
        model_name = self._extract_model_name(serialized)
        
        # Store run information
        self.active_runs[str(run_id)] = {
            "model_name": model_name,
            "prompts": prompts,
            "start_time": time.time(),
            "span_id": str(uuid.uuid4())[:16],
            "trace_id": str(uuid.uuid4())[:16]
        }
        
        logger.debug(f"LLM started: {model_name} for run {run_id}")
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        run_id = kwargs.get("run_id")
        if not run_id or str(run_id) not in self.active_runs:
            return
        
        run_data = self.active_runs[str(run_id)]
        end_time = time.time()
        latency_ms = int((end_time - run_data["start_time"]) * 1000)
        
        # Extract token usage and calculate costs
        token_usage = self._extract_token_usage(response)
        costs = self._calculate_costs(
            run_data["model_name"],
            token_usage["input_tokens"],
            token_usage["output_tokens"]
        )
        
        # Create OpenTelemetry span
        with self.tracer.start_as_current_span("llm_request") as span:
            span.set_attributes({
                "llm.model_name": run_data["model_name"],
                "llm.provider": run_data["model_name"].split("-")[0],
                "llm.application_name": self.application_name,
                "llm.user_id": self.user_id or "anonymous",
                "llm.session_id": self.session_id,
                "llm.input_tokens": token_usage["input_tokens"],
                "llm.output_tokens": token_usage["output_tokens"],
                "llm.total_tokens": token_usage["total_tokens"],
                "llm.prompt_cost_usd": costs["prompt_cost_usd"],
                "llm.completion_cost_usd": costs["completion_cost_usd"],
                "llm.total_cost_usd": costs["total_cost_usd"],
                "llm.latency_ms": latency_ms,
                "llm.prompts_count": len(run_data["prompts"]),
                "llm.responses_count": len(response.generations),
                "llm.timestamp": datetime.utcnow().isoformat()
            })
            
            # Store span data in database
            span_data = {
                "span_id": run_data["span_id"],
                "trace_id": run_data["trace_id"],
                "operation_name": "llm_request",
                "start_time": datetime.fromtimestamp(run_data["start_time"]),
                "end_time": datetime.fromtimestamp(end_time),
                "duration_ms": latency_ms,
                "status_code": 0,  # Success
                "attributes": dict(span.get_span_context())
            }
            
            # Store LLM metrics
            metrics_data = {
                "span_id": run_data["span_id"],
                "model_name": run_data["model_name"],
                "provider": run_data["model_name"].split("-")[0],
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "total_tokens": token_usage["total_tokens"],
                "prompt_cost_usd": costs["prompt_cost_usd"],
                "completion_cost_usd": costs["completion_cost_usd"],
                "total_cost_usd": costs["total_cost_usd"],
                "latency_ms": latency_ms,
                "application_name": self.application_name,
                "user_id": self.user_id,
                "session_id": self.session_id
            }
            
            try:
                # Store in database asynchronously
                await db_manager.store_span(span_data)
                await db_manager.store_llm_metrics(metrics_data)
                
                logger.info(
                    f"LLM request completed: {run_data['model_name']} "
                    f"${costs['total_cost_usd']:.6f} "
                    f"{token_usage['total_tokens']} tokens "
                    f"{latency_ms}ms"
                )
                
            except Exception as e:
                logger.error(f"Failed to store metrics: {e}")
        
        # Clean up
        del self.active_runs[str(run_id)]
    
    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        run_id = kwargs.get("run_id")
        if not run_id or str(run_id) not in self.active_runs:
            return
        
        run_data = self.active_runs[str(run_id)]
        end_time = time.time()
        latency_ms = int((end_time - run_data["start_time"]) * 1000)
        
        # Create error span
        with self.tracer.start_as_current_span("llm_request_error") as span:
            span.set_attributes({
                "llm.model_name": run_data["model_name"],
                "llm.application_name": self.application_name,
                "llm.user_id": self.user_id or "anonymous",
                "llm.error": str(error),
                "llm.latency_ms": latency_ms
            })
            
            span.record_exception(error)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        
        logger.error(f"LLM request failed: {run_data['model_name']} - {error}")
        
        # Clean up
        del self.active_runs[str(run_id)]


def create_cost_tracker(application_name: str = "default", user_id: Optional[str] = None) -> LLMCostTracker:
    """Factory function to create a cost tracker instance."""
    return LLMCostTracker(application_name=application_name, user_id=user_id)