"""OTLP data ingestion and PostgreSQL persistence."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import get_settings
from .database import get_database_pool

logger = logging.getLogger(__name__)

router = APIRouter()


class SpanData(BaseModel):
    """Span data model for PostgreSQL storage."""

    span_id: str = Field(..., max_length=32)
    trace_id: str = Field(..., max_length=32)
    parent_span_id: Optional[str] = Field(None, max_length=32)
    operation_name: str = Field(..., max_length=255)
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status_code: int = Field(default=0)
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class LLMMetricsData(BaseModel):
    """LLM-specific metrics extracted from span attributes."""

    span_id: str
    model_name: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    latency_ms: Optional[int] = None
    application_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class OTLPIngestionService:
    """Service for ingesting OTLP data and storing in PostgreSQL."""

    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.db_pool = await get_database_pool()
            logger.info("OTLP ingestion service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OTLP ingestion service: {e}")
            raise

    def extract_span_data(self, otlp_span: Dict[str, Any]) -> SpanData:
        """Extract span data from OTLP format."""
        # Convert OTLP timestamps (nanoseconds) to datetime
        start_time_ns = int(otlp_span.get("startTimeUnixNano", 0))
        end_time_ns = int(otlp_span.get("endTimeUnixNano", 0))

        start_time = (
            datetime.fromtimestamp(start_time_ns / 1_000_000_000)
            if start_time_ns
            else datetime.now()
        )
        end_time = (
            datetime.fromtimestamp(end_time_ns / 1_000_000_000) if end_time_ns else None
        )

        duration_ms = None
        if start_time and end_time:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Extract attributes
        attributes = {}
        for attr in otlp_span.get("attributes", []):
            key = attr.get("key", "")
            value = attr.get("value", {})
            if "stringValue" in value:
                attributes[key] = value["stringValue"]
            elif "intValue" in value:
                attributes[key] = int(value["intValue"])
            elif "doubleValue" in value:
                attributes[key] = float(value["doubleValue"])
            elif "boolValue" in value:
                attributes[key] = value["boolValue"]

        return SpanData(
            span_id=otlp_span.get("spanId", ""),
            trace_id=otlp_span.get("traceId", ""),
            parent_span_id=otlp_span.get("parentSpanId"),
            operation_name=otlp_span.get("name", "unknown"),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status_code=otlp_span.get("status", {}).get("code", 0),
            status_message=otlp_span.get("status", {}).get("message"),
            attributes=attributes,
        )

    def extract_llm_metrics(self, span_data: SpanData) -> Optional[LLMMetricsData]:
        """Extract LLM-specific metrics from span attributes."""
        attrs = span_data.attributes

        # Check if this is an LLM-related span
        model_name = attrs.get("llm.model.name") or attrs.get("gen_ai.request.model")
        if not model_name:
            return None

        provider = attrs.get("llm.provider") or attrs.get("gen_ai.system") or "unknown"

        # Extract token counts
        input_tokens = int(
            attrs.get("llm.usage.prompt_tokens", 0)
            or attrs.get("gen_ai.usage.prompt_tokens", 0)
        )
        output_tokens = int(
            attrs.get("llm.usage.completion_tokens", 0)
            or attrs.get("gen_ai.usage.completion_tokens", 0)
        )
        total_tokens = int(
            attrs.get("llm.usage.total_tokens", 0)
            or attrs.get("gen_ai.usage.total_tokens", 0)
        )

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        # Extract cost information
        prompt_cost = float(attrs.get("llm.cost.prompt_cost", 0))
        completion_cost = float(attrs.get("llm.cost.completion_cost", 0))
        total_cost = float(attrs.get("llm.cost.total_cost", 0))

        if total_cost == 0:
            total_cost = prompt_cost + completion_cost

        return LLMMetricsData(
            span_id=span_data.span_id,
            model_name=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_cost_usd=prompt_cost,
            completion_cost_usd=completion_cost,
            total_cost_usd=total_cost,
            latency_ms=span_data.duration_ms,
            application_name=attrs.get("service.name") or attrs.get("application.name"),
            user_id=attrs.get("user.id"),
            session_id=attrs.get("session.id"),
        )

    async def store_span(self, span_data: SpanData) -> bool:
        """Store span data in PostgreSQL."""
        if not self.db_pool:
            logger.error("Database pool not initialized")
            return False

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO spans (
                        span_id, trace_id, parent_span_id, operation_name,
                        start_time, end_time, duration_ms, status_code, 
                        status_message, attributes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (span_id) DO UPDATE SET
                        end_time = EXCLUDED.end_time,
                        duration_ms = EXCLUDED.duration_ms,
                        status_code = EXCLUDED.status_code,
                        status_message = EXCLUDED.status_message,
                        attributes = EXCLUDED.attributes
                """,
                    span_data.span_id,
                    span_data.trace_id,
                    span_data.parent_span_id,
                    span_data.operation_name,
                    span_data.start_time,
                    span_data.end_time,
                    span_data.duration_ms,
                    span_data.status_code,
                    span_data.status_message,
                    json.dumps(span_data.attributes),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store span {span_data.span_id}: {e}")
            return False

    async def store_llm_metrics(self, metrics_data: LLMMetricsData) -> bool:
        """Store LLM metrics in PostgreSQL."""
        if not self.db_pool:
            logger.error("Database pool not initialized")
            return False

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_metrics (
                        span_id, model_name, provider, input_tokens, output_tokens,
                        total_tokens, prompt_cost_usd, completion_cost_usd, 
                        total_cost_usd, latency_ms, application_name, user_id, session_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (span_id) DO UPDATE SET
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        prompt_cost_usd = EXCLUDED.prompt_cost_usd,
                        completion_cost_usd = EXCLUDED.completion_cost_usd,
                        total_cost_usd = EXCLUDED.total_cost_usd,
                        latency_ms = EXCLUDED.latency_ms
                """,
                    metrics_data.span_id,
                    metrics_data.model_name,
                    metrics_data.provider,
                    metrics_data.input_tokens,
                    metrics_data.output_tokens,
                    metrics_data.total_tokens,
                    metrics_data.prompt_cost_usd,
                    metrics_data.completion_cost_usd,
                    metrics_data.total_cost_usd,
                    metrics_data.latency_ms,
                    metrics_data.application_name,
                    metrics_data.user_id,
                    metrics_data.session_id,
                )
            return True
        except Exception as e:
            logger.error(
                f"Failed to store LLM metrics for span {metrics_data.span_id}: {e}"
            )
            return False

    async def process_otlp_traces(self, otlp_data: Dict[str, Any]) -> int:
        """Process OTLP traces data and store in PostgreSQL."""
        processed_spans = 0

        try:
            resource_spans = otlp_data.get("resourceSpans", [])

            for resource_span in resource_spans:
                scope_spans = resource_span.get("scopeSpans", [])

                for scope_span in scope_spans:
                    spans = scope_span.get("spans", [])

                    for otlp_span in spans:
                        # Extract and store span data
                        span_data = self.extract_span_data(otlp_span)
                        await self.store_span(span_data)

                        # Extract and store LLM metrics if applicable
                        llm_metrics = self.extract_llm_metrics(span_data)
                        if llm_metrics:
                            await self.store_llm_metrics(llm_metrics)

                        processed_spans += 1

            logger.info(f"Processed {processed_spans} spans from OTLP data")
            return processed_spans

        except Exception as e:
            logger.error(f"Failed to process OTLP traces: {e}")
            return processed_spans


# Global service instance
otlp_service = OTLPIngestionService()


@router.post("/v1/traces")
async def ingest_traces(request: Request) -> Response:
    """OTLP traces ingestion endpoint."""
    try:
        # Parse the request body
        content_type = request.headers.get("content-type", "")

        if "application/x-protobuf" in content_type:
            # For protobuf, we'd need to decode using protobuf libraries
            # For now, log and return success to keep the pipeline flowing
            body = await request.body()
            logger.info(f"Received protobuf OTLP data: {len(body)} bytes")
            return Response(status_code=200)

        elif "application/json" in content_type:
            # Handle JSON OTLP data
            otlp_data = await request.json()
            processed_spans = await otlp_service.process_otlp_traces(otlp_data)

            return Response(
                status_code=200, content=f"Processed {processed_spans} spans"
            )

        else:
            logger.warning(f"Unsupported content type: {content_type}")
            return Response(status_code=200)  # Accept anyway to avoid pipeline issues

    except Exception as e:
        logger.error(f"OTLP ingestion error: {e}")
        # Return 200 to avoid breaking the OTLP pipeline
        return Response(status_code=200)


@router.get("/v1/traces/health")
async def traces_health() -> Dict[str, Any]:
    """Health check for OTLP ingestion service."""
    return {
        "status": "healthy",
        "service": "otlp-ingestion",
        "database_connected": otlp_service.db_pool is not None,
    }


async def initialize_otlp_service():
    """Initialize the OTLP ingestion service."""
    await otlp_service.initialize()
