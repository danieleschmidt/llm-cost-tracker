"""User session data models."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UserSession(BaseModel):
    """User session tracking model."""
    
    session_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=100)
    application_name: Optional[str] = Field(None, max_length=100)
    start_time: datetime
    last_activity: datetime
    total_requests: int = Field(default=0, ge=0)
    total_cost_usd: Decimal = Field(default=Decimal("0.00"), ge=0)
    total_tokens: int = Field(default=0, ge=0)
    avg_latency_ms: float = Field(default=0.0, ge=0)
    models_used: List[str] = Field(default_factory=list)
    session_metadata: Optional[Dict] = Field(default_factory=dict)
    
    @property
    def session_duration_minutes(self) -> float:
        """Calculate session duration in minutes."""
        delta = self.last_activity - self.start_time
        return delta.total_seconds() / 60
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate requests per minute rate."""
        duration = self.session_duration_minutes
        if duration == 0:
            return 0.0
        return self.total_requests / duration
    
    @property
    def cost_per_request(self) -> Decimal:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return Decimal("0.00")
        return self.total_cost_usd / self.total_requests
    
    @property
    def tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests
    
    @property
    def is_active(self) -> bool:
        """Check if session is considered active (activity within 30 minutes)."""
        inactive_threshold = datetime.utcnow().timestamp() - (30 * 60)
        return self.last_activity.timestamp() > inactive_threshold
    
    @property
    def session_efficiency(self) -> str:
        """Categorize session efficiency."""
        cost_per_minute = float(self.total_cost_usd) / max(1, self.session_duration_minutes)
        
        if cost_per_minute < 0.01:
            return "efficient"
        elif cost_per_minute < 0.05:
            return "moderate"
        elif cost_per_minute < 0.20:
            return "expensive"
        else:
            return "premium"
    
    class Config:
        orm_mode = True


class SessionMetrics(BaseModel):
    """Session metrics aggregation model."""
    
    total_sessions: int = Field(default=0)
    active_sessions: int = Field(default=0)
    avg_session_duration_minutes: float = Field(default=0.0)
    avg_requests_per_session: float = Field(default=0.0)
    avg_cost_per_session: Decimal = Field(default=Decimal("0.00"))
    total_session_cost_usd: Decimal = Field(default=Decimal("0.00"))
    unique_users: int = Field(default=0)
    most_used_models: List[str] = Field(default_factory=list)
    peak_concurrent_sessions: int = Field(default=0)
    
    @property
    def user_engagement_score(self) -> float:
        """Calculate user engagement score (0-100)."""
        if self.total_sessions == 0:
            return 0.0
        
        # Factor in session duration, requests per session, and user retention
        duration_score = min(100, self.avg_session_duration_minutes * 2)
        activity_score = min(100, self.avg_requests_per_session * 10)
        retention_score = min(100, (self.total_sessions / max(1, self.unique_users)) * 20)
        
        return (duration_score + activity_score + retention_score) / 3