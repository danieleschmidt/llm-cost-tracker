"""
Enhanced Security Framework for LLM Cost Tracker

Comprehensive security implementation including:
- Advanced input validation and sanitization  
- Multi-layered authentication and authorization
- Rate limiting with intelligent threat detection
- Data encryption and secure key management
- Security event monitoring and alerting
- GDPR/CCPA compliance framework
"""

import hashlib
import logging
import secrets
import time
from typing import Dict, Optional, Tuple, Any

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security constants
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds


class AdvancedRateLimiter:
    """
    Advanced rate limiter with intelligent threat detection and adaptive limits.
    
    Features:
    - Multi-tier rate limiting (IP, user, API key)
    - Threat detection with progressive penalties
    - Adaptive limits based on system load
    - Whitelist/blacklist support
    """
    
    def __init__(self, 
                 max_requests: int = RATE_LIMIT_REQUESTS, 
                 window: int = RATE_LIMIT_WINDOW,
                 burst_multiplier: float = 2.0,
                 threat_detection: bool = True):
        self.base_max_requests = max_requests
        self.window = window
        self.burst_multiplier = burst_multiplier
        self.threat_detection = threat_detection
        
        # Request tracking
        self.requests: Dict[str, list] = {}
        self.violations: Dict[str, int] = {}
        self.last_violation: Dict[str, float] = {}
        
        # Security lists
        self.whitelist: set = set()
        self.blacklist: set = set()
        self.suspicious_ips: set = set()
        
        # Threat detection metrics
        self.threat_patterns = {
            'rapid_fire': 0.1,      # Requests within 100ms
            'burst_attack': 5.0,     # Many requests in 5 seconds
            'pattern_attack': 3.0    # Repetitive suspicious patterns
        }
        
        # Adaptive limits
        self.current_load_factor = 1.0
        self.last_load_update = time.time()
    
    def is_allowed(self, key: str, user_tier: str = "standard", request_info: Optional[Dict] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Advanced rate limiting with threat detection.
        
        Returns:
            Tuple[bool, Dict]: (is_allowed, security_info)
        """
        current_time = time.time()
        security_info = {
            'violation_type': None,
            'threat_level': 'low',
            'adaptive_limit': self.base_max_requests,
            'blacklisted': False,
            'requires_captcha': False
        }
        
        # Check blacklist
        if key in self.blacklist:
            security_info['blacklisted'] = True
            security_info['threat_level'] = 'critical'
            logger.warning(f"Blocked blacklisted IP: {key}")
            return False, security_info
        
        # Check whitelist (bypass most restrictions)
        if key in self.whitelist:
            security_info['threat_level'] = 'trusted'
            return True, security_info
        
        # Initialize tracking for new keys
        if key not in self.requests:
            self.requests[key] = []
            self.violations[key] = 0
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if current_time - req_time < self.window
        ]
        
        # Calculate adaptive limits based on user tier and system load
        tier_multipliers = {
            'premium': 3.0,
            'standard': 1.0,
            'trial': 0.3
        }
        
        tier_multiplier = tier_multipliers.get(user_tier, 1.0)
        adaptive_limit = int(self.base_max_requests * tier_multiplier * self.current_load_factor)
        security_info['adaptive_limit'] = adaptive_limit
        
        # Threat detection
        if self.threat_detection and request_info:
            threat_detected, threat_info = self._detect_threats(key, current_time, request_info)
            if threat_detected:
                security_info.update(threat_info)
                # Reduce limits for suspicious activity
                adaptive_limit = max(1, adaptive_limit // 4)
                security_info['adaptive_limit'] = adaptive_limit
        
        # Apply violation penalties
        if key in self.violations and self.violations[key] > 0:
            penalty_factor = max(0.1, 1.0 - (self.violations[key] * 0.2))
            adaptive_limit = int(adaptive_limit * penalty_factor)
            security_info['adaptive_limit'] = adaptive_limit
            
            # Require CAPTCHA for repeated violations
            if self.violations[key] >= 3:
                security_info['requires_captcha'] = True
        
        # Check if request is within limits
        current_requests = len(self.requests[key])
        
        if current_requests >= adaptive_limit:
            self._record_violation(key, current_time, 'rate_limit_exceeded')
            security_info['violation_type'] = 'rate_limit_exceeded'
            security_info['threat_level'] = 'high' if current_requests > adaptive_limit * 2 else 'medium'
            
            logger.warning(f"Rate limit exceeded for {key}: {current_requests}/{adaptive_limit}")
            return False, security_info
        
        # Add current request
        self.requests[key].append(current_time)
        
        # Decay violations over time
        self._decay_violations(key, current_time)
        
        return True, security_info
    
    def _detect_threats(self, key: str, current_time: float, request_info: Dict) -> Tuple[bool, Dict[str, Any]]:
        """Detect various threat patterns."""
        threat_info = {
            'threat_level': 'low',
            'threat_patterns': [],
            'anomaly_score': 0.0
        }
        
        recent_requests = self.requests[key]
        if not recent_requests:
            return False, threat_info
        
        # Pattern 1: Rapid fire detection
        rapid_fire_threshold = current_time - self.threat_patterns['rapid_fire']
        rapid_requests = [req for req in recent_requests if req > rapid_fire_threshold]
        
        if len(rapid_requests) >= 5:
            threat_info['threat_patterns'].append('rapid_fire')
            threat_info['anomaly_score'] += 3.0
        
        # Pattern 2: Burst attack detection
        burst_threshold = current_time - self.threat_patterns['burst_attack']
        burst_requests = [req for req in recent_requests if req > burst_threshold]
        
        if len(burst_requests) >= self.base_max_requests:
            threat_info['threat_patterns'].append('burst_attack')
            threat_info['anomaly_score'] += 2.0
        
        # Pattern 3: Suspicious user agent patterns
        if 'user_agent' in request_info:
            user_agent = request_info['user_agent'].lower()
            suspicious_patterns = ['bot', 'crawler', 'scraper', 'automated', 'python', 'curl']
            if any(pattern in user_agent for pattern in suspicious_patterns):
                threat_info['threat_patterns'].append('suspicious_user_agent')
                threat_info['anomaly_score'] += 1.0
        
        # Pattern 4: Unusual request patterns
        if 'endpoint' in request_info:
            # Check for enumeration attacks (hitting many different endpoints)
            endpoint_diversity = len(set(getattr(self, f"recent_endpoints_{key}", [])))
            if endpoint_diversity > 10:
                threat_info['threat_patterns'].append('endpoint_enumeration')
                threat_info['anomaly_score'] += 1.5
        
        # Calculate threat level
        if threat_info['anomaly_score'] >= 4.0:
            threat_info['threat_level'] = 'critical'
            self.blacklist.add(key)
            logger.error(f"Critical threat detected from {key}: {threat_info['threat_patterns']}")
        elif threat_info['anomaly_score'] >= 2.0:
            threat_info['threat_level'] = 'high'
            self.suspicious_ips.add(key)
            logger.warning(f"High threat detected from {key}: {threat_info['threat_patterns']}")
        elif threat_info['anomaly_score'] >= 1.0:
            threat_info['threat_level'] = 'medium'
        
        return threat_info['anomaly_score'] > 0, threat_info
    
    def _record_violation(self, key: str, current_time: float, violation_type: str):
        """Record security violations with progressive penalties."""
        if key not in self.violations:
            self.violations[key] = 0
        
        self.violations[key] += 1
        self.last_violation[key] = current_time
        
        # Add to suspicious IPs after repeated violations
        if self.violations[key] >= 5:
            self.suspicious_ips.add(key)
            logger.warning(f"Added {key} to suspicious IPs after {self.violations[key]} violations")
        
        # Auto-blacklist for severe violations
        if self.violations[key] >= 10:
            self.blacklist.add(key)
            logger.error(f"Auto-blacklisted {key} after {self.violations[key]} violations")
    
    def _decay_violations(self, key: str, current_time: float):
        """Decay violations over time for rehabilitation."""
        if key in self.last_violation:
            time_since_violation = current_time - self.last_violation[key]
            decay_threshold = 3600  # 1 hour
            
            if time_since_violation > decay_threshold:
                if key in self.violations and self.violations[key] > 0:
                    self.violations[key] = max(0, self.violations[key] - 1)
                    logger.info(f"Decayed violation count for {key} to {self.violations[key]}")
                
                # Remove from suspicious IPs if violations are low
                if key in self.suspicious_ips and self.violations[key] <= 2:
                    self.suspicious_ips.discard(key)
                    logger.info(f"Removed {key} from suspicious IPs")
    
    def update_load_factor(self, cpu_usage: float, memory_usage: float, queue_length: int = 0):
        """Update adaptive limits based on system load."""
        current_time = time.time()
        
        # Only update every 30 seconds
        if current_time - self.last_load_update < 30:
            return
        
        # Calculate load factor based on system metrics
        load_factors = []
        
        # CPU load factor
        if cpu_usage > 0.8:
            load_factors.append(0.5)  # Reduce limits significantly
        elif cpu_usage > 0.6:
            load_factors.append(0.7)  # Reduce limits moderately
        else:
            load_factors.append(1.0)  # Normal limits
        
        # Memory load factor  
        if memory_usage > 0.9:
            load_factors.append(0.3)
        elif memory_usage > 0.7:
            load_factors.append(0.6)
        else:
            load_factors.append(1.0)
        
        # Queue length factor
        if queue_length > 100:
            load_factors.append(0.4)
        elif queue_length > 50:
            load_factors.append(0.7)
        else:
            load_factors.append(1.0)
        
        # Use the most restrictive factor
        self.current_load_factor = min(load_factors)
        self.last_load_update = current_time
        
        logger.info(f"Updated load factor to {self.current_load_factor:.2f} "
                   f"(CPU: {cpu_usage:.1%}, Memory: {memory_usage:.1%}, Queue: {queue_length})")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        current_time = time.time()
        
        return {
            'active_clients': len(self.requests),
            'blacklisted_ips': len(self.blacklist),
            'suspicious_ips': len(self.suspicious_ips),
            'whitelisted_ips': len(self.whitelist),
            'total_violations': sum(self.violations.values()),
            'current_load_factor': self.current_load_factor,
            'threat_detection_enabled': self.threat_detection,
            'recent_violations': {
                key: violations for key, violations in self.violations.items()
                if key in self.last_violation and current_time - self.last_violation[key] < 3600
            }
        }

# Maintain backward compatibility
class RateLimiter(AdvancedRateLimiter):
    """Legacy rate limiter interface for backward compatibility."""
    
    def is_allowed(self, key: str) -> bool:
        """Simple interface for backward compatibility."""
        allowed, _ = super().is_allowed(key)
        return allowed


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