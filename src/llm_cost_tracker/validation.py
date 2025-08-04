"""Enhanced input validation and sanitization."""

import re
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, ValidationError
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class SecurityValidationError(ValidationError):
    """Security-related validation error."""
    pass


def validate_string(
    value: Any,
    min_length: int = 0,
    max_length: int = 1000,
    pattern: Optional[str] = None,
    allow_empty: bool = True,
    field_name: str = "field"
) -> str:
    """
    Validate and sanitize string input.
    
    Args:
        value: Input value to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        pattern: Regex pattern to match (optional)
        allow_empty: Whether empty strings are allowed
        field_name: Name of field for error messages
    
    Returns:
        Validated and sanitized string
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        if value is None and allow_empty:
            return ""
        raise ValidationError(f"{field_name} must be a string, got {type(value)}")
    
    # Check length
    if len(value) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")
    
    if len(value) > max_length:
        raise ValidationError(f"{field_name} must not exceed {max_length} characters")
    
    # Check if empty when not allowed
    if not allow_empty and not value.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    # Check pattern if provided
    if pattern and not re.match(pattern, value):
        raise ValidationError(f"{field_name} does not match required pattern")
    
    # Sanitize dangerous characters (but preserve allowed ones)
    sanitized = sanitize_string_content(value)
    
    return sanitized


def validate_integer(
    value: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    field_name: str = "field"
) -> int:
    """
    Validate integer input.
    
    Args:
        value: Input value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Name of field for error messages
    
    Returns:
        Validated integer
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be an integer")
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name} must be at least {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} must not exceed {max_value}")
    
    return value


def validate_decimal(
    value: Any,
    min_value: Optional[Decimal] = None,
    max_value: Optional[Decimal] = None,
    max_digits: int = 10,
    decimal_places: int = 6,
    field_name: str = "field"
) -> Decimal:
    """
    Validate decimal input.
    
    Args:
        value: Input value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        max_digits: Maximum total digits
        decimal_places: Maximum decimal places
        field_name: Name of field for error messages
    
    Returns:
        Validated Decimal
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValidationError(f"{field_name} must be a valid decimal number")
    
    # Check bounds
    if min_value is not None and decimal_value < min_value:
        raise ValidationError(f"{field_name} must be at least {min_value}")
    
    if max_value is not None and decimal_value > max_value:
        raise ValidationError(f"{field_name} must not exceed {max_value}")
    
    # Check precision
    sign, digits, exponent = decimal_value.as_tuple()
    
    if len(digits) > max_digits:
        raise ValidationError(f"{field_name} has too many digits (max {max_digits})")
    
    if exponent < -decimal_places:
        raise ValidationError(f"{field_name} has too many decimal places (max {decimal_places})")
    
    return decimal_value


def validate_datetime(
    value: Any,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    field_name: str = "field"
) -> datetime:
    """
    Validate datetime input.
    
    Args:
        value: Input value to validate
        min_date: Minimum allowed date
        max_date: Maximum allowed date
        field_name: Name of field for error messages
    
    Returns:
        Validated datetime
    
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, str):
        try:
            # Try to parse ISO format
            dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            raise ValidationError(f"{field_name} must be a valid ISO datetime string")
    else:
        raise ValidationError(f"{field_name} must be a datetime or ISO string")
    
    if min_date and dt_value < min_date:
        raise ValidationError(f"{field_name} cannot be before {min_date}")
    
    if max_date and dt_value > max_date:
        raise ValidationError(f"{field_name} cannot be after {max_date}")
    
    return dt_value


def validate_uuid(value: Any, field_name: str = "field") -> str:
    """
    Validate UUID input.
    
    Args:
        value: Input value to validate
        field_name: Name of field for error messages
    
    Returns:
        Validated UUID string
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    
    try:
        # This will raise ValueError if invalid
        uuid.UUID(value)
        return value
    except ValueError:
        raise ValidationError(f"{field_name} must be a valid UUID")


def validate_span_id(value: Any, field_name: str = "span_id") -> str:
    """Validate OpenTelemetry span ID format."""
    validated = validate_string(
        value,
        min_length=1,
        max_length=16,
        pattern=r'^[a-f0-9]+$',
        allow_empty=False,
        field_name=field_name
    )
    return validated


def validate_trace_id(value: Any, field_name: str = "trace_id") -> str:
    """Validate OpenTelemetry trace ID format."""
    validated = validate_string(
        value,
        min_length=1,
        max_length=32,
        pattern=r'^[a-f0-9]+$',
        allow_empty=False,
        field_name=field_name
    )
    return validated


def sanitize_string_content(value: str) -> str:
    """
    Sanitize string content to prevent injection attacks.
    
    Args:
        value: String to sanitize
    
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove null bytes and control characters (except common whitespace)
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    
    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())
    
    return sanitized


def validate_json_object(
    value: Any,
    max_depth: int = 10,
    max_keys: int = 100,
    field_name: str = "field"
) -> Dict[str, Any]:
    """
    Validate JSON object input.
    
    Args:
        value: Input value to validate
        max_depth: Maximum nesting depth
        max_keys: Maximum number of keys
        field_name: Name of field for error messages
    
    Returns:
        Validated dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be a dictionary")
    
    def check_depth(obj, current_depth=0):
        if current_depth > max_depth:
            raise ValidationError(f"{field_name} exceeds maximum nesting depth of {max_depth}")
        
        if isinstance(obj, dict):
            if len(obj) > max_keys:
                raise ValidationError(f"{field_name} has too many keys (max {max_keys})")
            
            for key, val in obj.items():
                if not isinstance(key, str):
                    raise ValidationError(f"{field_name} keys must be strings")
                check_depth(val, current_depth + 1)
        
        elif isinstance(obj, list):
            if len(obj) > max_keys:  # Apply same limit to arrays
                raise ValidationError(f"{field_name} array too large (max {max_keys} items)")
            
            for item in obj:
                check_depth(item, current_depth + 1)
    
    check_depth(value)
    return value


class SpanDataValidator(BaseModel):
    """Validator for span data input."""
    
    span_id: str = Field(..., min_length=1, max_length=16)
    trace_id: str = Field(..., min_length=1, max_length=32)
    parent_span_id: Optional[str] = Field(None, max_length=16)
    operation_name: str = Field(..., min_length=1, max_length=255)
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = Field(None, ge=0, le=3600000)  # Max 1 hour
    status_code: Optional[int] = Field(0, ge=0, le=2)
    status_message: Optional[str] = Field(None, max_length=1000)
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('span_id')
    def validate_span_id_format(cls, v):
        return validate_span_id(v)
    
    @validator('trace_id')
    def validate_trace_id_format(cls, v):
        return validate_trace_id(v)
    
    @validator('parent_span_id')
    def validate_parent_span_id_format(cls, v):
        if v is not None:
            return validate_span_id(v, "parent_span_id")
        return v
    
    @validator('operation_name')
    def validate_operation_name(cls, v):
        return sanitize_string_content(v)
    
    @validator('attributes')
    def validate_attributes_dict(cls, v):
        if v is not None:
            return validate_json_object(v, max_depth=5, max_keys=50, field_name="attributes")
        return {}


class LLMMetricsValidator(BaseModel):
    """Validator for LLM metrics input."""
    
    model_name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=50)
    input_tokens: int = Field(..., ge=0, le=1000000)
    output_tokens: int = Field(..., ge=0, le=1000000)
    total_cost_usd: Decimal = Field(..., ge=0, max_digits=10, decimal_places=6)
    latency_ms: Optional[int] = Field(None, ge=0, le=300000)  # Max 5 minutes
    application_name: Optional[str] = Field(None, max_length=100)
    user_id: Optional[str] = Field(None, max_length=100)
    session_id: Optional[str] = Field(None, max_length=255)
    prompt_text: Optional[str] = Field(None, max_length=10000)
    response_text: Optional[str] = Field(None, max_length=50000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('model_name', 'provider')
    def validate_string_fields(cls, v):
        return sanitize_string_content(v)
    
    @validator('application_name', 'user_id', 'session_id')
    def validate_optional_string_fields(cls, v):
        if v is not None:
            return sanitize_string_content(v)
        return v
    
    @validator('prompt_text', 'response_text')
    def validate_text_fields(cls, v):
        if v is not None:
            # More permissive sanitization for text content
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
        return v
    
    @validator('metadata')
    def validate_metadata_dict(cls, v):
        if v is not None:
            return validate_json_object(v, max_depth=3, max_keys=20, field_name="metadata")
        return {}


def validate_span_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate span data using the validator."""
    try:
        validator = SpanDataValidator(**data)
        return validator.dict()
    except ValidationError as e:
        logger.error(f"Span data validation failed: {e}")
        raise ValidationError(f"Invalid span data: {e}")


def validate_llm_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate LLM metrics data using the validator."""
    try:
        validator = LLMMetricsValidator(**data)
        return validator.dict()
    except ValidationError as e:
        logger.error(f"LLM metrics validation failed: {e}")
        raise ValidationError(f"Invalid metrics data: {e}")


def security_scan_input(data: Any, field_name: str = "input") -> None:
    """
    Perform security scanning on input data.
    
    Args:
        data: Data to scan
        field_name: Name of field for error messages
    
    Raises:
        SecurityValidationError: If security issues are found
    """
    if isinstance(data, str):
        # Check for common injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript\s*:',              # JavaScript protocol
            r'on\w+\s*=',                  # Event handlers
            r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL injection
            r'system\s*\(',                # System calls
            r'eval\s*\(',                  # Code evaluation
            r'exec\s*\(',                  # Code execution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                logger.warning(f"Security threat detected in {field_name}: {pattern}")
                raise SecurityValidationError(f"Potential security threat detected in {field_name}")
    
    elif isinstance(data, dict):
        for key, value in data.items():
            security_scan_input(key, f"{field_name}.key")
            security_scan_input(value, f"{field_name}.{key}")
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            security_scan_input(item, f"{field_name}[{i}]")