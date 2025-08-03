"""Cost record data models."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from pydantic import BaseModel, Field


class CostRecord(BaseModel):
    """Individual cost record model."""
    
    id: int
    trace_id: str = Field(..., min_length=1, max_length=32)
    span_id: str = Field(..., min_length=1, max_length=16)
    timestamp: datetime
    application_name: Optional[str] = Field(None, max_length=100)
    user_id: Optional[str] = Field(None, max_length=100)
    model_name: str = Field(..., max_length=100)
    provider: str = Field(..., max_length=50)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_cost_usd: Decimal = Field(..., ge=0, decimal_places=6)
    latency_ms: int = Field(..., ge=0)
    metadata: Optional[Dict] = Field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_per_token(self) -> Decimal:
        """Calculate cost per token."""
        if self.total_tokens == 0:
            return Decimal("0.00")
        return self.total_cost_usd / self.total_tokens
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second throughput."""
        if self.latency_ms == 0:
            return 0.0
        return float(self.total_tokens / (self.latency_ms / 1000))
    
    class Config:
        orm_mode = True


class CostSummary(BaseModel):
    """Cost summary aggregation model."""
    
    total_cost_usd: Decimal = Field(default=Decimal("0.00"))
    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    avg_cost_per_request: Decimal = Field(default=Decimal("0.00"))
    avg_cost_per_token: Decimal = Field(default=Decimal("0.00"))
    unique_models: int = Field(default=0)
    unique_users: int = Field(default=0)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    
    @property
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score (0-100)."""
        if self.total_cost_usd == 0:
            return 100.0
        
        # Simple efficiency metric based on cost per token and latency
        # Lower cost per token and lower latency = higher efficiency
        cost_factor = min(100, float(1 / (self.avg_cost_per_token * 1000000)) if self.avg_cost_per_token > 0 else 100)
        latency_factor = min(100, 10000 / max(1, self.avg_latency_ms))
        
        return (cost_factor + latency_factor) / 2