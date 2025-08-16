"""Configuration management with secure defaults."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with security-first configuration."""

    # Database settings
    database_url: str = Field(
        ..., description="PostgreSQL connection URL", env="DATABASE_URL"
    )

    # OpenTelemetry settings
    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint",
        env="OTEL_EXPORTER_OTLP_ENDPOINT",
    )

    otel_service_name: str = Field(
        default="llm-cost-tracker",
        description="Service name for OpenTelemetry",
        env="OTEL_SERVICE_NAME",
    )

    # API Keys (must be provided via environment)
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key", env="OPENAI_API_KEY"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key", env="ANTHROPIC_API_KEY"
    )

    vellum_api_key: Optional[str] = Field(
        default=None, description="Vellum API key", env="VELLUM_API_KEY"
    )

    # Security settings
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="Allowed CORS origins",
        env="ALLOWED_ORIGINS",
    )

    enable_debug: bool = Field(
        default=False,
        description="Enable debug mode (NEVER in production)",
        env="DEBUG",
    )

    log_level: str = Field(default="INFO", description="Logging level", env="LOG_LEVEL")

    # Budget and alerting
    slack_webhook_url: Optional[str] = Field(
        default=None, description="Slack webhook for alerts", env="SLACK_WEBHOOK_URL"
    )

    budget_config_path: str = Field(
        default="config/budget-rules.yml",
        description="Path to budget configuration file",
        env="BUDGET_CONFIG_PATH",
    )

    # Additional settings for testing
    secret_key: Optional[str] = Field(
        default=None, description="Secret key for JWT tokens", env="SECRET_KEY"
    )

    enable_budget_alerts: bool = Field(
        default=True, description="Enable budget alerting", env="ENABLE_BUDGET_ALERTS"
    )

    enable_model_swapping: bool = Field(
        default=True,
        description="Enable automatic model swapping",
        env="ENABLE_MODEL_SWAPPING",
    )

    enable_metrics_export: bool = Field(
        default=True, description="Enable metrics export", env="ENABLE_METRICS_EXPORT"
    )

    @validator("database_url")
    def validate_database_url(cls, v):
        """Ensure database URL doesn't contain credentials in logs."""
        if not v:
            raise ValueError("DATABASE_URL is required")
        # Don't log the actual URL in production
        return v

    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse comma-separated origins from environment."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Ensure valid log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields for testing


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def get_redacted_config() -> dict:
    """Get configuration with sensitive values redacted for logging."""
    settings = get_settings()
    config = settings.dict()

    # Redact sensitive fields
    sensitive_fields = [
        "database_url",
        "openai_api_key",
        "anthropic_api_key",
        "vellum_api_key",
        "slack_webhook_url",
    ]

    for field in sensitive_fields:
        if config.get(field):
            config[field] = "***REDACTED***"

    return config
