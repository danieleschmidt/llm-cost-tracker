"""Database models for sentiment analysis with enhanced validation and security."""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import Column, String, Float, DateTime, Integer, Text, JSON, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import validates
from sqlalchemy.sql import func

from .database import Base
from .sentiment_analyzer import SentimentLabel
from .validation import validate_text_input, ValidationError
from .security import anonymize_sensitive_data

class SentimentAnalysisRequest(Base):
    """Database model for sentiment analysis requests with audit trail."""
    
    __tablename__ = "sentiment_analysis_requests"
    
    # Primary key and identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # User and session tracking
    user_id = Column(String(128), nullable=True, index=True)  # Anonymized user ID
    session_id = Column(String(64), nullable=True, index=True)
    ip_address_hash = Column(String(128), nullable=True)  # Hashed IP for security
    
    # Request content (anonymized)
    text_hash = Column(String(128), nullable=False, index=True)  # Hash for duplicate detection
    text_length = Column(Integer, nullable=False)
    language = Column(String(5), nullable=False, default="en")
    model_requested = Column(String(64), nullable=True)
    
    # Results
    sentiment_label = Column(ENUM(SentimentLabel), nullable=True)
    confidence_score = Column(Float, nullable=True)
    sentiment_scores = Column(JSON, nullable=True)  # Detailed scores for all labels
    
    # Performance and cost metrics
    processing_time_ms = Column(Float, nullable=True)
    model_used = Column(String(64), nullable=True)
    cost_usd = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Error handling
    error_type = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps and lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Compliance and data governance
    data_retention_until = Column(DateTime(timezone=True), nullable=True)
    is_anonymized = Column(String(10), default="true", nullable=False)  # GDPR compliance
    consent_given = Column(String(10), default="false", nullable=False)  # Explicit consent
    processing_purpose = Column(String(128), default="sentiment_analysis")
    
    # Quality and validation flags
    validation_passed = Column(String(10), default="false", nullable=False)
    security_scan_passed = Column(String(10), default="false", nullable=False)
    is_batch_request = Column(String(10), default="false", nullable=False)
    batch_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Performance monitoring
    cache_hit = Column(String(10), default="false", nullable=False)
    circuit_breaker_triggered = Column(String(10), default="false", nullable=False)
    quantum_planning_used = Column(String(10), default="false", nullable=False)
    
    # Database constraints
    __table_args__ = (
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0", name="valid_confidence"),
        CheckConstraint("processing_time_ms >= 0", name="positive_processing_time"),
        CheckConstraint("cost_usd >= 0", name="positive_cost"),
        CheckConstraint("tokens_used >= 0", name="positive_tokens"),
        CheckConstraint("text_length >= 1 AND text_length <= 100000", name="valid_text_length"),
        CheckConstraint("retry_count >= 0 AND retry_count <= 5", name="valid_retry_count"),
        Index("idx_sentiment_created_at", "created_at"),
        Index("idx_sentiment_user_created", "user_id", "created_at"),
        Index("idx_sentiment_model_language", "model_used", "language"),
        Index("idx_sentiment_performance", "processing_time_ms", "cost_usd"),
        Index("idx_sentiment_quality", "validation_passed", "security_scan_passed"),
        Index("idx_sentiment_batch", "batch_id", "is_batch_request"),
    )
    
    @validates('text_length')
    def validate_text_length(self, key, value):
        """Validate text length constraints."""
        if value is None:
            raise ValidationError("Text length cannot be null")
        if not isinstance(value, int) or value < 1 or value > 100000:
            raise ValidationError("Text length must be between 1 and 100000 characters")
        return value
    
    @validates('language')
    def validate_language(self, key, value):
        """Validate language code format."""
        if value is None:
            return "en"  # Default to English
        if not isinstance(value, str) or len(value) != 2 or not value.isalpha():
            raise ValidationError("Language must be a 2-letter ISO code")
        return value.lower()
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, value):
        """Validate confidence score range."""
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                raise ValidationError("Confidence score must be between 0.0 and 1.0")
        return value
    
    @validates('processing_time_ms')
    def validate_processing_time(self, key, value):
        """Validate processing time is non-negative."""
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValidationError("Processing time must be non-negative")
        return value
    
    @validates('cost_usd')
    def validate_cost(self, key, value):
        """Validate cost is non-negative."""
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValidationError("Cost must be non-negative")
        return value
    
    @validates('retry_count')
    def validate_retry_count(self, key, value):
        """Validate retry count is within acceptable range."""
        if value is not None:
            if not isinstance(value, int) or value < 0 or value > 5:
                raise ValidationError("Retry count must be between 0 and 5")
        return value
    
    @validates('user_id')
    def validate_user_id(self, key, value):
        """Validate and anonymize user ID."""
        if value is not None:
            # Anonymize sensitive user data
            return anonymize_sensitive_data(str(value))
        return value
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        base_data = {
            "id": str(self.id),
            "request_id": self.request_id,
            "text_length": self.text_length,
            "language": self.language,
            "model_requested": self.model_requested,
            "sentiment_label": self.sentiment_label.value if self.sentiment_label else None,
            "confidence_score": self.confidence_score,
            "sentiment_scores": self.sentiment_scores,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "cost_usd": self.cost_usd,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cache_hit": self.cache_hit == "true",
            "quantum_planning_used": self.quantum_planning_used == "true",
            "is_batch_request": self.is_batch_request == "true"
        }
        
        # Include sensitive data only if explicitly requested (for admin/audit purposes)
        if include_sensitive:
            base_data.update({
                "user_id": self.user_id,
                "session_id": self.session_id,
                "ip_address_hash": self.ip_address_hash,
                "text_hash": self.text_hash,
                "error_type": self.error_type,
                "error_message": self.error_message,
                "retry_count": self.retry_count,
                "validation_passed": self.validation_passed == "true",
                "security_scan_passed": self.security_scan_passed == "true",
                "data_retention_until": self.data_retention_until.isoformat() if self.data_retention_until else None,
                "consent_given": self.consent_given == "true",
                "processing_purpose": self.processing_purpose
            })
        
        return base_data
    
    @classmethod
    def create_from_request(
        cls,
        request_id: str,
        text: str,
        language: str = "en",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        model_requested: Optional[str] = None,
        is_batch: bool = False,
        batch_id: Optional[str] = None,
        consent_given: bool = False
    ) -> "SentimentAnalysisRequest":
        """Create a new sentiment analysis request record."""
        import hashlib
        
        # Validate input
        validate_text_input(text, min_length=1, max_length=100000)
        
        # Create hashes for privacy
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        ip_hash = hashlib.sha256(ip_address.encode('utf-8')).hexdigest() if ip_address else None
        
        # Calculate data retention (30 days for non-consented, 1 year for consented)
        retention_days = 365 if consent_given else 30
        data_retention_until = datetime.utcnow() + datetime.timedelta(days=retention_days)
        
        return cls(
            request_id=request_id,
            user_id=anonymize_sensitive_data(user_id) if user_id else None,
            session_id=session_id,
            ip_address_hash=ip_hash,
            text_hash=text_hash,
            text_length=len(text),
            language=language.lower() if language else "en",
            model_requested=model_requested,
            is_batch_request="true" if is_batch else "false",
            batch_id=batch_id,
            consent_given="true" if consent_given else "false",
            data_retention_until=data_retention_until,
            validation_passed="false",  # Will be updated after validation
            security_scan_passed="false"  # Will be updated after security scan
        )
    
    def update_result(
        self,
        sentiment_label: SentimentLabel,
        confidence_score: float,
        sentiment_scores: Dict[str, float],
        processing_time_ms: float,
        model_used: str,
        cost_usd: float = None,
        tokens_used: int = None,
        cache_hit: bool = False,
        quantum_planning_used: bool = False
    ):
        """Update with analysis results."""
        self.sentiment_label = sentiment_label
        self.confidence_score = confidence_score
        self.sentiment_scores = sentiment_scores
        self.processing_time_ms = processing_time_ms
        self.model_used = model_used
        self.cost_usd = cost_usd
        self.tokens_used = tokens_used
        self.cache_hit = "true" if cache_hit else "false"
        self.quantum_planning_used = "true" if quantum_planning_used else "false"
        self.completed_at = datetime.utcnow()
        self.validation_passed = "true"  # Successful completion implies validation passed
    
    def update_error(
        self,
        error_type: str,
        error_message: str,
        retry_count: int = 0
    ):
        """Update with error information."""
        self.error_type = error_type
        self.error_message = error_message[:1000]  # Truncate long error messages
        self.retry_count = retry_count
        self.completed_at = datetime.utcnow()
    
    def mark_security_validated(self, passed: bool = True):
        """Mark security validation status."""
        self.security_scan_passed = "true" if passed else "false"
    
    def __repr__(self):
        return (
            f"<SentimentAnalysisRequest(id={self.id}, "
            f"request_id={self.request_id}, "
            f"sentiment={self.sentiment_label}, "
            f"confidence={self.confidence_score})>"
        )