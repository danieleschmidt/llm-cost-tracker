"""Budget rule data models."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, validator


class BudgetRuleBase(BaseModel):
    """Base budget rule model."""
    
    name: str = Field(..., min_length=1, max_length=100)
    monthly_limit_usd: Decimal = Field(..., gt=0, decimal_places=2)
    alert_threshold: Decimal = Field(default=Decimal("0.8"), ge=0, le=1)
    auto_switch_enabled: bool = Field(default=False)
    fallback_model: Optional[str] = Field(None, max_length=100)
    
    @validator('monthly_limit_usd')
    def validate_monthly_limit(cls, v):
        if v <= 0:
            raise ValueError('Monthly limit must be positive')
        if v > 1_000_000:
            raise ValueError('Monthly limit cannot exceed $1,000,000')
        return v
    
    @validator('fallback_model')
    def validate_fallback_model(cls, v, values):
        if values.get('auto_switch_enabled') and not v:
            raise ValueError('Fallback model required when auto switch is enabled')
        return v


class BudgetRuleCreate(BudgetRuleBase):
    """Budget rule creation model."""
    pass


class BudgetRuleUpdate(BaseModel):
    """Budget rule update model."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    monthly_limit_usd: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    alert_threshold: Optional[Decimal] = Field(None, ge=0, le=1)
    auto_switch_enabled: Optional[bool] = None
    fallback_model: Optional[str] = Field(None, max_length=100)


class BudgetRule(BudgetRuleBase):
    """Complete budget rule model."""
    
    id: int
    current_spend_usd: Decimal = Field(default=Decimal("0.00"))
    created_at: datetime
    updated_at: datetime
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization percentage."""
        if self.monthly_limit_usd == 0:
            return 0.0
        return float(self.current_spend_usd / self.monthly_limit_usd * 100)
    
    @property
    def is_over_threshold(self) -> bool:
        """Check if spending is over alert threshold."""
        return self.current_spend_usd >= (self.monthly_limit_usd * self.alert_threshold)
    
    @property
    def is_over_budget(self) -> bool:
        """Check if spending exceeds monthly limit."""
        return self.current_spend_usd >= self.monthly_limit_usd
    
    @property
    def remaining_budget_usd(self) -> Decimal:
        """Calculate remaining budget."""
        return max(Decimal("0.00"), self.monthly_limit_usd - self.current_spend_usd)
    
    class Config:
        orm_mode = True