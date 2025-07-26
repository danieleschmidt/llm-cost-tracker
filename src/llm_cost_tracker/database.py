"""Database operations for LLM metrics storage."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
from asyncpg import Pool

from .config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                get_settings().database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                logger.info("Database connection test successful")
                
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def store_span(self, span_data: Dict) -> str:
        """Store OpenTelemetry span data."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        query = """
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
        RETURNING span_id
        """
        
        try:
            async with self.pool.acquire() as conn:
                span_id = await conn.fetchval(
                    query,
                    span_data["span_id"],
                    span_data["trace_id"],
                    span_data.get("parent_span_id"),
                    span_data["operation_name"],
                    span_data["start_time"],
                    span_data.get("end_time"),
                    span_data.get("duration_ms"),
                    span_data.get("status_code", 0),
                    span_data.get("status_message"),
                    span_data.get("attributes", {})
                )
                return span_id
                
        except Exception as e:
            logger.error("Failed to store span: %s", e)
            raise
    
    async def store_llm_metrics(self, metrics_data: Dict) -> str:
        """Store LLM-specific metrics."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        query = """
        INSERT INTO llm_metrics (
            span_id, model_name, provider, input_tokens, output_tokens,
            total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            latency_ms, application_name, user_id, session_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
        """
        
        try:
            async with self.pool.acquire() as conn:
                metric_id = await conn.fetchval(
                    query,
                    metrics_data.get("span_id"),
                    metrics_data["model_name"],
                    metrics_data["provider"],
                    metrics_data.get("input_tokens", 0),
                    metrics_data.get("output_tokens", 0),
                    metrics_data.get("total_tokens", 0),
                    metrics_data.get("prompt_cost_usd", 0.0),
                    metrics_data.get("completion_cost_usd", 0.0),
                    metrics_data.get("total_cost_usd", 0.0),
                    metrics_data.get("latency_ms"),
                    metrics_data.get("application_name"),
                    metrics_data.get("user_id"),
                    metrics_data.get("session_id")
                )
                return str(metric_id)
                
        except Exception as e:
            logger.error("Failed to store LLM metrics: %s", e)
            raise
    
    async def update_budget_usage(self, application_name: str, user_id: Optional[str], 
                                cost: float, period_start: datetime, period_end: datetime) -> None:
        """Update budget usage for application/user."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        query = """
        INSERT INTO budget_usage (
            application_name, user_id, period_start, period_end, total_cost_usd, updated_at
        ) VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (application_name, user_id, period_start, period_end) DO UPDATE SET
            total_cost_usd = budget_usage.total_cost_usd + EXCLUDED.total_cost_usd,
            updated_at = NOW()
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    application_name,
                    user_id,
                    period_start,
                    period_end,
                    cost
                )
                
        except Exception as e:
            logger.error("Failed to update budget usage: %s", e)
            raise
    
    async def get_budget_usage(self, application_name: str, user_id: Optional[str],
                             period_start: datetime, period_end: datetime) -> Optional[Dict]:
        """Get current budget usage."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        query = """
        SELECT total_cost_usd, budget_limit_usd, updated_at
        FROM budget_usage
        WHERE application_name = $1 
        AND ($2 IS NULL OR user_id = $2)
        AND period_start = $3 AND period_end = $4
        """
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                    application_name,
                    user_id,
                    period_start,
                    period_end
                )
                
                if row:
                    return {
                        "total_cost_usd": float(row["total_cost_usd"]),
                        "budget_limit_usd": float(row["budget_limit_usd"]) if row["budget_limit_usd"] else None,
                        "updated_at": row["updated_at"]
                    }
                return None
                
        except Exception as e:
            logger.error("Failed to get budget usage: %s", e)
            raise
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get metrics summary for the specified time period."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        query = """
        SELECT 
            COUNT(*) as total_requests,
            SUM(total_cost_usd) as total_cost,
            AVG(latency_ms) as avg_latency,
            COUNT(DISTINCT model_name) as unique_models,
            COUNT(DISTINCT application_name) as unique_applications
        FROM llm_metrics
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        """
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, hours)
                
                return {
                    "total_requests": row["total_requests"] or 0,
                    "total_cost": float(row["total_cost"] or 0),
                    "avg_latency": float(row["avg_latency"] or 0),
                    "unique_models": row["unique_models"] or 0,
                    "unique_applications": row["unique_applications"] or 0,
                    "period_hours": hours
                }
                
        except Exception as e:
            logger.error("Failed to get metrics summary: %s", e)
            raise


# Global database manager instance
db_manager = DatabaseManager()