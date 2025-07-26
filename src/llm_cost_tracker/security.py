"""Security utilities and middleware."""

import hashlib
import logging
import secrets
import time
from typing import Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security constants
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed based on rate limits."""
        current_time = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if current_time - req_time < self.window
        ]
        
        # Check if under limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(current_time)
        return True


class APIKeyAuth(HTTPBearer):
    """API key authentication for secure endpoints."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__(auto_error=False)
        self.api_keys = api_keys or {}
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key from request."""
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            return None
        
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        
        # Hash the provided key for comparison
        provided_key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        
        # Check against known API key hashes
        for key_name, key_hash in self.api_keys.items():
            if secrets.compare_digest(provided_key_hash, key_hash):
                logger.info(f"Authenticated request with key: {key_name}")
                return key_name
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


def sanitize_user_input(value: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    
    if len(value) > max_length:
        raise ValueError(f"Input too long (max {max_length} characters)")
    
    # Remove potentially dangerous characters
    # Keep alphanumeric, spaces, and common punctuation
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_@")
    sanitized = "".join(c for c in value if c in allowed_chars)
    
    return sanitized.strip()


def mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive data in dictionaries for logging."""
    sensitive_keys = {
        "api_key", "password", "secret", "token", "key", 
        "authorization", "x-api-key", "database_url"
    }
    
    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
        else:
            masked[key] = value
    
    return masked


async def validate_request_size(request: Request):
    """Middleware to validate request size."""
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large (max {MAX_REQUEST_SIZE} bytes)"
            )


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


class SecurityHeaders:
    """Security headers middleware."""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response