"""Main FastAPI application for LLM Cost Tracker."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .alert_webhooks import router as alert_router
from .config import get_redacted_config, get_settings
from .database import db_manager
from .otlp_ingestion import initialize_otlp_service, router as otlp_router
from .security import RateLimiter, SecurityHeaders, validate_request_size

# Configure logging
logging.basicConfig(
    level=getattr(logging, get_settings().log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting LLM Cost Tracker")
    logger.info("Configuration: %s", get_redacted_config())
    
    # Initialize database
    await db_manager.initialize()
    
    # Initialize OTLP ingestion service
    await initialize_otlp_service()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Cost Tracker")
    await db_manager.close()


app = FastAPI(
    title="LLM Cost Tracker",
    description="OpenTelemetry-based cost tracking for LLM applications",
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


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Apply security middleware to all requests."""
    # Validate request size
    await validate_request_size(request)
    
    # Apply rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    
    # Process request
    response = await call_next(request)
    
    # Add security headers
    response = SecurityHeaders.add_security_headers(response)
    
    return response


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy", 
        "service": "llm-cost-tracker",
        "version": "0.1.0"
    })


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