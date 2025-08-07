"""Main FastAPI application for LLM Cost Tracker."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .alert_webhooks import router as alert_router
from .config import get_redacted_config, get_settings
from .database import db_manager, DatabaseConnectionError, DatabaseOperationError
from .otlp_ingestion import initialize_otlp_service, router as otlp_router
from .security import RateLimiter, SecurityHeaders, validate_request_size
from .logging_config import configure_logging, set_request_context, generate_request_id, get_logger
from .health_checks import health_checker
from .circuit_breaker import circuit_registry
from .validation import ValidationError, SecurityValidationError
from .cache import llm_cache
from .concurrency import task_queue
from .auto_scaling import auto_scaler, metrics_collector, start_auto_scaling

# Configure structured logging
try:
    configure_logging(get_settings().log_level, structured=True)
    logger = get_logger(__name__)
except Exception:
    # Fallback to basic logging if structured logging fails
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events with enhanced error handling."""
    request_id = generate_request_id()
    set_request_context(request_id=request_id, user_id="system")
    
    # Startup
    logger.info("Starting LLM Cost Tracker", extra={"startup": True})
    logger.info("Configuration loaded", extra={"config": get_redacted_config()})
    
    # Initialize database with proper error handling
    try:
        await db_manager.initialize()
        logger.info("Database initialized successfully")
    except DatabaseConnectionError as e:
        logger.error(f"Database connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        raise
    
    # Initialize OTLP ingestion service
    try:
        await initialize_otlp_service()
        logger.info("OTLP service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OTLP service: {e}")
        # Don't fail startup for OTLP issues
    
    # Start auto-scaling system
    try:
        await start_auto_scaling()
        logger.info("Auto-scaling system started")
    except Exception as e:
        logger.error(f"Failed to start auto-scaling: {e}")
        # Don't fail startup for auto-scaling issues
    
    # Start task queue workers
    try:
        await task_queue.start_workers()
        logger.info("Task queue workers started")
    except Exception as e:
        logger.error(f"Failed to start task queue: {e}")
    
    logger.info("Application startup completed", extra={"startup_complete": True})
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Cost Tracker", extra={"shutdown": True})
    
    # Shutdown task queue
    try:
        await task_queue.shutdown()
        logger.info("Task queue shutdown completed")
    except Exception as e:
        logger.error(f"Error during task queue shutdown: {e}")
    
    # Stop metrics collection
    try:
        metrics_collector.stop_collection()
        logger.info("Metrics collection stopped")
    except Exception as e:
        logger.error(f"Error stopping metrics collection: {e}")
    
    # Clear cache
    try:
        await llm_cache.clear()
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
    
    # Close database connections
    try:
        await db_manager.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")
    
    logger.info("Application shutdown completed", extra={"shutdown_complete": True})


app = FastAPI(
    title="Sentiment Analyzer Pro with LLM Cost Tracking",
    description="Advanced sentiment analysis with quantum-enhanced task planning and OpenTelemetry-based cost tracking",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if get_settings().enable_debug else None,
    redoc_url="/redoc" if get_settings().enable_debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(otlp_router, tags=["otlp"])
app.include_router(alert_router, tags=["alerts"])

# Add error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Validation error: {exc}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "request_id": request_id
        }
    )

@app.exception_handler(SecurityValidationError)
async def security_validation_error_handler(request: Request, exc: SecurityValidationError):
    """Handle security validation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Security validation error: {exc}", extra={"request_id": request_id, "security_threat": True})
    return JSONResponse(
        status_code=400,
        content={
            "error": "Security Validation Error", 
            "message": "Input contains potentially dangerous content",
            "request_id": request_id
        }
    )

@app.exception_handler(DatabaseConnectionError)
async def database_connection_error_handler(request: Request, exc: DatabaseConnectionError):
    """Handle database connection errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Database connection error: {exc}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": "Database connection issue. Please try again later.",
            "request_id": request_id
        }
    )

@app.exception_handler(DatabaseOperationError)
async def database_operation_error_handler(request: Request, exc: DatabaseOperationError):
    """Handle database operation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Database operation error: {exc}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Database operation failed. Please try again later.",
            "request_id": request_id
        }
    )

@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Request validation error: {exc}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request Validation Error",
            "message": "Invalid request format",
            "details": exc.errors(),
            "request_id": request_id
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Internal server error: {exc}", extra={"request_id": request_id}, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id
        }
    )

# Include new API routers
from .controllers import budget_router, cost_router, session_router, quantum_router
from .controllers.sentiment_controller import router as sentiment_router
app.include_router(budget_router, tags=["budget"])
app.include_router(cost_router, tags=["cost-analysis"])
app.include_router(session_router, tags=["sessions"])
app.include_router(quantum_router, tags=["quantum-planning"])
app.include_router(sentiment_router, tags=["sentiment-analysis"])


@app.middleware("http")
async def enhanced_security_middleware(request: Request, call_next):
    """Enhanced security middleware with request tracking."""
    # Generate request ID and set context
    request_id = generate_request_id()
    request.state.request_id = request_id
    
    # Extract user ID from headers if available
    user_id = request.headers.get("x-user-id", "anonymous")
    set_request_context(request_id=request_id, user_id=user_id)
    
    start_time = datetime.utcnow()
    
    try:
        # Validate request size
        await validate_request_size(request)
        
        # Apply rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}", extra={
                "client_ip": client_ip,
                "path": str(request.url.path),
                "method": request.method
            })
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests", 
                    "message": "Rate limit exceeded",
                    "request_id": request_id
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response = SecurityHeaders.add_security_headers(response)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        # Log request completion
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Request completed", extra={
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "duration_ms": duration,
            "client_ip": client_ip,
            "user_id": user_id
        })
        
        return response
        
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Request failed: {e}", extra={
            "method": request.method,
            "path": str(request.url.path),
            "duration_ms": duration,
            "client_ip": request.client.host if request.client else "unknown",
            "user_id": user_id
        }, exc_info=True)
        raise


@app.get("/health")
async def health_check() -> JSONResponse:
    """Basic health check endpoint for load balancers."""
    return JSONResponse({
        "status": "healthy", 
        "service": "llm-cost-tracker",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.get("/health/detailed")
async def detailed_health_check() -> JSONResponse:
    """Detailed health check with all subsystem status."""
    try:
        health_status = await health_checker.get_health_status()
        return JSONResponse(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check system failure",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/health/circuit-breakers")
async def circuit_breaker_status() -> JSONResponse:
    """Get status of all circuit breakers."""
    try:
        states = circuit_registry.get_all_states()
        return JSONResponse({
            "circuit_breakers": states,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Circuit breaker status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get circuit breaker status",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/metrics/cache")
async def cache_metrics() -> JSONResponse:
    """Get cache performance metrics."""
    try:
        stats = await llm_cache.get_stats()
        return JSONResponse({
            "cache_stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Cache metrics failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get cache metrics",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/metrics/concurrency")
async def concurrency_metrics() -> JSONResponse:
    """Get concurrency and task queue metrics."""
    try:
        stats = task_queue.get_stats()
        return JSONResponse({
            "task_queue_stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Concurrency metrics failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get concurrency metrics",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/metrics/auto-scaling")
async def auto_scaling_metrics() -> JSONResponse:
    """Get auto-scaling configuration and history."""
    try:
        config = auto_scaler.get_current_configuration()
        history = auto_scaler.get_scaling_history(limit=20)
        recent_metrics = metrics_collector.get_recent_metrics(seconds=300)
        
        return JSONResponse({
            "configuration": config,
            "scaling_history": history,
            "recent_metrics_count": len(recent_metrics),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Auto-scaling metrics failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get auto-scaling metrics",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """Readiness check endpoint for Kubernetes."""
    try:
        # Check database connection
        db_healthy = await db_manager.check_health()
        
        if not db_healthy:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "service": "llm-cost-tracker",
                    "reason": "database_unavailable"
                }
            )
        
        return JSONResponse({
            "status": "ready",
            "service": "llm-cost-tracker",
            "version": "0.1.0",
            "checks": {
                "database": "healthy"
            }
        })
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "llm-cost-tracker",
                "error": str(e)
            }
        )


@app.get("/health/live")
async def liveness_check() -> JSONResponse:
    """Liveness check endpoint for Kubernetes."""
    return JSONResponse({
        "status": "alive",
        "service": "llm-cost-tracker",
        "version": "0.1.0",
        "uptime_seconds": app.state.uptime if hasattr(app.state, 'uptime') else 0
    })


@app.get("/metrics")
async def metrics_endpoint() -> JSONResponse:
    """Prometheus metrics endpoint."""
    try:
        # Get basic metrics summary
        metrics_summary = await db_manager.get_metrics_summary()
        
        return JSONResponse({
            "service": "llm-cost-tracker",
            "metrics": metrics_summary,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "metrics_unavailable",
                "message": str(e)
            }
        )


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint."""
    return JSONResponse({
        "message": "LLM Cost Tracker API",
        "version": "0.1.0",
        "docs": "/docs" if get_settings().enable_debug else "Documentation disabled in production"
    })


@app.get("/config")
async def get_config() -> JSONResponse:
    """Get redacted configuration (debug only)."""
    if not get_settings().enable_debug:
        return JSONResponse(
            status_code=404,
            content={"error": "Not found"}
        )
    
    return JSONResponse(get_redacted_config())