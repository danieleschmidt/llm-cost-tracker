"""Tests for security utilities."""

import pytest
from fastapi import HTTPException

from llm_cost_tracker.security import (
    RateLimiter, APIKeyAuth, sanitize_user_input, 
    mask_sensitive_data, generate_api_key, hash_api_key
)


def test_rate_limiter():
    """Test rate limiting functionality."""
    limiter = RateLimiter(max_requests=2, window=60)
    key = "test_client"
    
    # First two requests should be allowed
    assert limiter.is_allowed(key) is True
    assert limiter.is_allowed(key) is True
    
    # Third request should be blocked
    assert limiter.is_allowed(key) is False


def test_sanitize_user_input():
    """Test input sanitization."""
    # Valid input
    result = sanitize_user_input("Hello world! How are you?")
    assert result == "Hello world! How are you?"
    
    # Input with dangerous characters
    result = sanitize_user_input("Hello <script>alert('xss')</script>")
    assert "<script>" not in result
    assert ">" not in result
    assert "<" not in result
    assert "(" not in result
    assert ")" not in result
    # The word "alert" will still be there as it contains allowed characters
    
    # Too long input
    with pytest.raises(ValueError):
        sanitize_user_input("x" * 1001)


def test_mask_sensitive_data():
    """Test sensitive data masking."""
    data = {
        "api_key": "secret123",
        "username": "user",
        "password": "pass123",
        "normal_field": "visible"
    }
    
    masked = mask_sensitive_data(data)
    
    assert masked["api_key"] == "***MASKED***"
    assert masked["password"] == "***MASKED***"
    assert masked["username"] == "user"
    assert masked["normal_field"] == "visible"


def test_api_key_generation():
    """Test API key generation and hashing."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    
    # Keys should be different
    assert key1 != key2
    
    # Keys should be URL-safe
    assert all(c.isalnum() or c in '-_' for c in key1)
    
    # Hashing should be consistent
    hash1 = hash_api_key(key1)
    hash2 = hash_api_key(key1)
    assert hash1 == hash2
    
    # Different keys should have different hashes
    hash3 = hash_api_key(key2)
    assert hash1 != hash3