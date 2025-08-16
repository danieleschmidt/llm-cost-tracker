"""Database operations for LLM metrics storage."""

import asyncio
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional

import asyncpg
from asyncpg import Pool
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.types import Numeric

from .cache import cache_result, llm_cache
from .concurrency import TaskPriority, task_queue
from .config import get_settings

logger = logging.getLogger(__name__)


def retry_db_operation(max_retries: int = 3, backoff_factor: float = 0.5):
    """Decorator to retry database operations with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (
                    asyncpg.ConnectionDoesNotExistError,
                    asyncpg.InterfaceError,
                    asyncpg.PostgresConnectionError,
                ) as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Database operation failed after {max_retries} retries: {e}"
                        )
                        break

                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # Don't retry for other types of exceptions
                    logger.error(
                        f"Database operation failed with non-retryable error: {e}"
                    )
                    raise

            raise last_exception

        return wrapper

    return decorator


class DatabaseError(Exception):
    """Custom database error for better error handling."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection related errors."""

    pass


class DatabaseOperationError(DatabaseError):
    """Database operation related errors."""

    pass


# SQLAlchemy Base for ORM models
Base = declarative_base()


class SpanModel(Base):
    """SQLAlchemy model for spans table."""

    __tablename__ = "spans"

    span_id = Column(String(32), primary_key=True)
    trace_id = Column(String(32), nullable=False, index=True)
    parent_span_id = Column(String(32))
    operation_name = Column(String(255), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    status_code = Column(Integer, default=0)
    status_message = Column(Text)
    attributes = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationship to LLM metrics
    llm_metric = relationship("LLMMetricModel", back_populates="span", uselist=False)


class LLMMetricModel(Base):
    """SQLAlchemy model for llm_metrics table."""

    __tablename__ = "llm_metrics"

    id = Column(UUID, primary_key=True, server_default=func.uuid_generate_v4())
    span_id = Column(String(32), ForeignKey("spans.span_id"), unique=True)
    model_name = Column(String(255), nullable=False, index=True)
    provider = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    prompt_cost_usd = Column(Numeric(10, 6), default=0)
    completion_cost_usd = Column(Numeric(10, 6), default=0)
    total_cost_usd = Column(Numeric(10, 6), default=0)
    latency_ms = Column(Integer)
    application_name = Column(String(255), index=True)
    user_id = Column(String(255))
    session_id = Column(String(255))
    timestamp = Column(DateTime(timezone=True), default=func.now(), index=True)

    # Relationship to span
    span = relationship("SpanModel", back_populates="llm_metric")


class UsageLogModel(Base):
    """SQLAlchemy model for llm_usage_logs table."""

    __tablename__ = "llm_usage_logs"

    id = Column(Integer, primary_key=True)
    trace_id = Column(String(32), nullable=False, index=True)
    span_id = Column(String(16), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=func.now(), index=True)
    application_name = Column(String(100), index=True)
    user_id = Column(String(100), index=True)
    model_name = Column(String(100), nullable=False, index=True)
    provider = Column(String(50), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Numeric(10, 6), default=0)
    latency_ms = Column(Integer, default=0)
    prompt_text = Column(Text)
    response_text = Column(Text)
    request_metadata = Column("metadata", JSONB, default={})


class BudgetRuleModel(Base):
    """SQLAlchemy model for budget_rules table."""

    __tablename__ = "budget_rules"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    monthly_limit_usd = Column(Numeric(10, 2), nullable=False)
    current_spend_usd = Column(Numeric(10, 2), default=0)
    alert_threshold = Column(Numeric(3, 2), default=0.8)
    auto_switch_enabled = Column(Boolean, default=False, index=True)
    fallback_model = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )


class UserSessionModel(Base):
    """SQLAlchemy model for user_sessions table."""

    __tablename__ = "user_sessions"

    session_id = Column(String(64), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    application_name = Column(String(100))
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    last_activity = Column(DateTime(timezone=True), nullable=False, index=True)
    total_requests = Column(Integer, default=0)
    total_cost_usd = Column(Numeric(10, 6), default=0)
    total_tokens = Column(Integer, default=0)
    avg_latency_ms = Column(Numeric(10, 2), default=0)
    models_used = Column(Text, default="[]")
    session_metadata = Column(Text, default="{}")


# Async database session factory
async_engine = None
AsyncSessionLocal = None


async def get_async_engine():
    """Get or create async database engine."""
    global async_engine
    if async_engine is None:
        database_url = get_settings().database_url
        # Convert sync postgres URL to async if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        async_engine = create_async_engine(
            database_url,
            echo=get_settings().log_level == "DEBUG",
            future=True,
            pool_size=10,
            max_overflow=20,
        )
    return async_engine


async def get_async_session():
    """Get async database session."""
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        engine = await get_async_engine()
        AsyncSessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return AsyncSessionLocal()


async def get_db() -> AsyncSession:
    """Dependency for getting database session."""
    async with await get_async_session() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self):
        self.pool: Optional[Pool] = None

    async def initialize(self) -> None:
        """Initialize database connection pool with enhanced error handling."""
        try:
            database_url = get_settings().database_url
            if not database_url:
                raise DatabaseConnectionError("DATABASE_URL not configured")

            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                max_queries=50,
                max_inactive_connection_lifetime=300,
            )
            logger.info("Database connection pool initialized successfully")

            # Test connection with retry
            await self._test_connection()

        except asyncpg.InvalidCatalogNameError as e:
            raise DatabaseConnectionError(f"Database does not exist: {e}")
        except asyncpg.InvalidPasswordError as e:
            raise DatabaseConnectionError(f"Invalid database credentials: {e}")
        except asyncpg.ConnectionDoesNotExistError as e:
            raise DatabaseConnectionError(f"Cannot connect to database: {e}")
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise DatabaseConnectionError(f"Database initialization failed: {e}")

    @retry_db_operation(max_retries=3)
    async def _test_connection(self) -> None:
        """Test database connection with retry mechanism."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def check_health(self) -> bool:
        """Check database health for readiness probe."""
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @retry_db_operation(max_retries=3)
    async def store_span(self, span_data: Dict) -> str:
        """Store OpenTelemetry span data with enhanced error handling."""
        if not self.pool:
            raise DatabaseConnectionError("Database not initialized")

        # Validate required fields
        required_fields = ["span_id", "trace_id", "operation_name", "start_time"]
        for field in required_fields:
            if field not in span_data:
                raise DatabaseOperationError(f"Missing required field: {field}")

        # Validate span_id format
        if not isinstance(span_data["span_id"], str) or len(span_data["span_id"]) > 32:
            raise DatabaseOperationError("Invalid span_id format")

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
                    span_data.get("attributes", {}),
                )
                logger.debug(f"Successfully stored span: {span_id}")
                return span_id

        except asyncpg.UniqueViolationError as e:
            logger.warning(f"Span already exists, updating: {span_data['span_id']}")
            raise DatabaseOperationError(f"Span already exists: {e}")
        except asyncpg.DataError as e:
            logger.error(f"Invalid data for span storage: {e}")
            raise DatabaseOperationError(f"Invalid span data: {e}")
        except Exception as e:
            logger.error("Failed to store span: %s", e)
            raise DatabaseOperationError(f"Span storage failed: {e}")

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
                    metrics_data.get("session_id"),
                )
                return str(metric_id)

        except Exception as e:
            logger.error("Failed to store LLM metrics: %s", e)
            raise

    async def update_budget_usage(
        self,
        application_name: str,
        user_id: Optional[str],
        cost: float,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
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
                    query, application_name, user_id, period_start, period_end, cost
                )

        except Exception as e:
            logger.error("Failed to update budget usage: %s", e)
            raise

    async def get_budget_usage(
        self,
        application_name: str,
        user_id: Optional[str],
        period_start: datetime,
        period_end: datetime,
    ) -> Optional[Dict]:
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
                    query, application_name, user_id, period_start, period_end
                )

                if row:
                    return {
                        "total_cost_usd": float(row["total_cost_usd"]),
                        "budget_limit_usd": (
                            float(row["budget_limit_usd"])
                            if row["budget_limit_usd"]
                            else None
                        ),
                        "updated_at": row["updated_at"],
                    }
                return None

        except Exception as e:
            logger.error("Failed to get budget usage: %s", e)
            raise

    async def get_metrics_summary(self, hours: int = 24, user_id: str = None) -> Dict:
        """Get metrics summary for the specified time period with caching."""
        if not self.pool:
            raise DatabaseConnectionError("Database not initialized")

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
                    "period_hours": hours,
                }

        except Exception as e:
            logger.error("Failed to get metrics summary: %s", e)
            raise


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


async def get_database_pool() -> Optional[Pool]:
    """Get the database connection pool."""
    if not db_manager.pool:
        await db_manager.initialize()
    return db_manager.pool
