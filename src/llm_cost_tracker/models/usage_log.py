"""Usage log data models."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from pydantic import BaseModel, Field


class UsageLogBase(BaseModel):
    """Base usage log model."""
    
    trace_id: str = Field(..., min_length=1, max_length=32)
    span_id: str = Field(..., min_length=1, max_length=16)
    application_name: Optional[str] = Field(None, max_length=100)
    user_id: Optional[str] = Field(None, max_length=100)
    model_name: str = Field(..., max_length=100)
    provider: str = Field(..., max_length=50)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_cost_usd: Decimal = Field(..., ge=0, decimal_places=6)
    latency_ms: int = Field(..., ge=0)
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    metadata: Optional[Dict] = Field(default_factory=dict)


class UsageLogCreate(UsageLogBase):
    """Usage log creation model."""
    pass


class UsageLog(UsageLogBase):
    """Complete usage log model."""
    
    id: int
    timestamp: datetime
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return self.input_tokens + self.output_tokens
    
    @property
    def input_output_ratio(self) -> float:
        """Calculate input to output token ratio."""
        if self.output_tokens == 0:
            return float('inf') if self.input_tokens > 0 else 0.0
        return self.input_tokens / self.output_tokens
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on tokens per ms."""
        if self.latency_ms == 0:
            return 0.0
        return self.total_tokens / self.latency_ms
    
    @property
    def cost_category(self) -> str:
        """Categorize cost level."""
        cost = float(self.total_cost_usd)
        if cost < 0.001:
            return "micro"
        elif cost < 0.01:
            return "low"
        elif cost < 0.1:
            return "medium"
        elif cost < 1.0:
            return "high"
        else:
            return "premium"
    
    class Config:
        orm_mode = True