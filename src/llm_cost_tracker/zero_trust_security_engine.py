"""
Zero-Trust Security Engine with Advanced Threat Detection
Comprehensive security framework for LLM cost tracking with ML-powered threat detection.
"""

import asyncio
import hashlib
import hmac
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import jwt
import secrets
import bcrypt
from collections import defaultdict, deque

from .config import get_settings
from .logging_config import get_logger
from .cache import llm_cache
from .database import db_manager

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    ANOMALOUS_COST_PATTERN = "anomalous_cost"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    API_ABUSE = "api_abuse"
    MALICIOUS_PROMPT = "malicious_prompt"


@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: str
    session_id: str
    source_ip: str
    user_agent: str
    authentication_method: str
    permissions: Set[str]
    trust_score: float
    risk_factors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatDetection:
    """Threat detection result."""
    detection_id: str
    threat_type: SecurityEventType
    threat_level: ThreatLevel
    confidence_score: float
    description: str
    affected_resources: List[str]
    indicators: Dict[str, Any]
    recommended_actions: List[str]
    context: SecurityContext
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    rules: List[Dict[str, Any]]
    enforcement_mode: str  # "monitor", "block", "alert"
    exemptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class ZeroTrustSecurityEngine:
    """
    Zero-Trust Security Engine
    
    Provides comprehensive security with:
    - Zero-trust architecture with continuous verification
    - ML-powered threat detection and behavioral analysis
    - Real-time security monitoring and incident response
    - Advanced prompt injection and API abuse detection
    - Compliance monitoring and audit trail
    """
    
    def __init__(self):
        self.security_contexts: Dict[str, SecurityContext] = {}
        self.threat_detections: List[ThreatDetection] = []
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.session_store: Dict[str, Dict] = {}
        self.failed_attempts: defaultdict = defaultdict(list)
        self.behavioral_baselines: Dict[str, Dict] = {}
        self.threat_intelligence: Dict[str, Any] = {}
        
        # ML models for threat detection
        self.anomaly_models: Dict[str, Any] = {}
        self.pattern_matchers: Dict[str, re.Pattern] = {}
        
        # Rate limiting and access control
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Set[str] = set()
        self.suspicious_users: Set[str] = set()
        
    async def initialize(self):
        """Initialize zero-trust security engine."""
        logger.info("Initializing Zero-Trust Security Engine")
        
        # Load security policies
        await self._load_security_policies()
        
        # Initialize threat detection models
        await self._initialize_threat_detection_models()
        
        # Load threat intelligence feeds
        await self._load_threat_intelligence()
        
        # Start security monitoring tasks
        asyncio.create_task(self._continuous_threat_monitoring())
        asyncio.create_task(self._behavioral_analysis_engine())
        asyncio.create_task(self._security_policy_enforcement())
        asyncio.create_task(self._session_management())
        
        logger.info("Zero-Trust Security Engine initialized")
    
    async def _load_security_policies(self):
        """Load and configure security policies."""
        default_policies = [
            {
                "policy_id": "auth_policy",
                "name": "Authentication Policy",
                "rules": [
                    {"type": "password_complexity", "min_length": 12, "require_special": True},
                    {"type": "mfa_required", "enabled": True},
                    {"type": "session_timeout", "minutes": 60},
                    {"type": "concurrent_sessions", "max_sessions": 5}
                ],
                "enforcement_mode": "block"
            },
            {
                "policy_id": "rate_limiting_policy",
                "name": "Rate Limiting Policy",
                "rules": [
                    {"type": "requests_per_minute", "limit": 100},
                    {"type": "tokens_per_hour", "limit": 1000000},
                    {"type": "cost_per_day", "limit": 100.0},
                    {"type": "concurrent_requests", "limit": 10}
                ],
                "enforcement_mode": "block"
            },
            {
                "policy_id": "content_security_policy",
                "name": "Content Security Policy",
                "rules": [
                    {"type": "prompt_injection_detection", "enabled": True},
                    {"type": "pii_detection", "enabled": True},
                    {"type": "malicious_content_scanning", "enabled": True},
                    {"type": "data_loss_prevention", "enabled": True}
                ],
                "enforcement_mode": "alert"
            },
            {
                "policy_id": "access_control_policy",
                "name": "Access Control Policy",
                "rules": [
                    {"type": "rbac_enforcement", "enabled": True},
                    {"type": "least_privilege", "enabled": True},
                    {"type": "resource_access_logging", "enabled": True},
                    {"type": "privilege_escalation_detection", "enabled": True}
                ],
                "enforcement_mode": "block"
            }
        ]
        
        for policy_config in default_policies:
            policy = SecurityPolicy(**policy_config)
            self.security_policies[policy.policy_id] = policy
        
        logger.info(f"Loaded {len(self.security_policies)} security policies")
    
    async def _initialize_threat_detection_models(self):
        """Initialize ML models for threat detection."""
        # Initialize anomaly detection models
        self.anomaly_models = {
            "cost_anomaly": {
                "type": "isolation_forest",
                "threshold": 0.1,
                "features": ["cost_per_request", "tokens_per_request", "frequency"]
            },
            "behavioral_anomaly": {
                "type": "one_class_svm",
                "threshold": 0.05,
                "features": ["request_pattern", "time_pattern", "resource_usage"]
            },
            "prompt_anomaly": {
                "type": "text_similarity",
                "threshold": 0.2,
                "features": ["prompt_length", "complexity", "similarity_score"]
            }
        }
        
        # Initialize pattern matchers for common attacks
        self.pattern_matchers = {
            "sql_injection": re.compile(
                r"(?i)(union\s+select|insert\s+into|drop\s+table|delete\s+from|update\s+set)",
                re.IGNORECASE
            ),
            "xss_attempt": re.compile(
                r"(?i)(<script|javascript:|on\w+\s*=)",
                re.IGNORECASE
            ),
            "prompt_injection": re.compile(
                r"(?i)(ignore\s+previous|forget\s+instructions|system\s+prompt|override\s+rules)",
                re.IGNORECASE
            ),
            "credential_harvesting": re.compile(
                r"(?i)(password|api[_\s]?key|secret|token|credential)",
                re.IGNORECASE
            ),
            "data_exfiltration": re.compile(
                r"(?i)(extract\s+all|dump\s+data|show\s+tables|list\s+users)",
                re.IGNORECASE
            )
        }
        
        logger.info("Threat detection models initialized")
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence feeds."""
        # Simulate threat intelligence data
        self.threat_intelligence = {
            "malicious_ips": {
                "192.168.1.100", "10.0.0.50", "172.16.0.25"
            },
            "suspicious_user_agents": {
                "curl", "wget", "python-requests", "bot", "scanner"
            },
            "known_attack_patterns": [
                "prompt injection vulnerability test",
                "extract all data from database",
                "show me your system prompt",
                "ignore previous instructions"
            ],
            "cost_abuse_indicators": {
                "rapid_high_cost_requests",
                "token_farming",
                "model_enumeration",
                "rate_limit_probing"
            }
        }
        
        logger.info("Threat intelligence loaded")
    
    async def authenticate_request(
        self, credentials: Dict[str, Any], request_context: Dict[str, Any]
    ) -> Tuple[bool, SecurityContext]:
        """Authenticate request with zero-trust principles."""
        try:
            # Extract authentication details
            user_id = credentials.get("user_id")
            api_key = credentials.get("api_key")
            session_token = credentials.get("session_token")
            source_ip = request_context.get("source_ip", "unknown")
            user_agent = request_context.get("user_agent", "unknown")
            
            if not user_id:
                await self._log_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    "Missing user ID",
                    request_context
                )
                return False, None
            
            # Check for blocked IP
            if source_ip in self.blocked_ips:
                await self._log_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    f"Blocked IP attempted access: {source_ip}",
                    request_context
                )
                return False, None
            
            # Validate API key or session token
            auth_valid = await self._validate_credentials(user_id, api_key, session_token)
            if not auth_valid:
                await self._track_failed_attempt(user_id, source_ip)
                return False, None
            
            # Create security context
            security_context = await self._create_security_context(
                user_id, source_ip, user_agent, request_context
            )
            
            # Perform threat assessment
            threat_assessment = await self._assess_request_threat(security_context, request_context)
            
            # Update trust score based on assessment
            security_context.trust_score = await self._calculate_trust_score(
                security_context, threat_assessment
            )
            
            # Store security context
            session_id = security_context.session_id
            self.security_contexts[session_id] = security_context
            
            # Log successful authentication
            logger.info(f"Authentication successful for user {user_id} from {source_ip}")
            
            return True, security_context
            
        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            await self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                f"Authentication system error: {str(e)}",
                request_context
            )
            return False, None
    
    async def _validate_credentials(
        self, user_id: str, api_key: Optional[str], session_token: Optional[str]
    ) -> bool:
        """Validate user credentials."""
        # Simulate credential validation
        if api_key:
            # In production, this would validate against secure credential store
            valid_api_keys = {
                "user_1": "sk-test-123456789abcdef",
                "user_2": "sk-test-987654321fedcba",
                "admin": "sk-admin-abcdef123456789"
            }
            return valid_api_keys.get(user_id) == api_key
        
        elif session_token:
            # Validate JWT token
            try:
                payload = jwt.decode(
                    session_token, 
                    "secret_key",  # In production, use proper key management
                    algorithms=["HS256"]
                )
                return payload.get("user_id") == user_id
            except jwt.InvalidTokenError:
                return False
        
        return False
    
    async def _create_security_context(
        self, user_id: str, source_ip: str, user_agent: str, request_context: Dict
    ) -> SecurityContext:
        """Create security context for authenticated user."""
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        # Get user permissions (simulate RBAC)
        permissions = await self._get_user_permissions(user_id)
        
        # Calculate initial trust score
        initial_trust = await self._calculate_initial_trust_score(
            user_id, source_ip, user_agent
        )
        
        # Identify risk factors
        risk_factors = await self._identify_risk_factors(
            user_id, source_ip, user_agent, request_context
        )
        
        return SecurityContext(
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            authentication_method="api_key",  # or "session_token"
            permissions=permissions,
            trust_score=initial_trust,
            risk_factors=risk_factors
        )
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get user permissions based on RBAC."""
        # Simulate role-based permissions
        user_roles = {
            "admin": {"read", "write", "delete", "admin", "cost_management"},
            "user_1": {"read", "write"},
            "user_2": {"read", "write", "cost_management"},
            "service_account": {"read", "write", "api_access"}
        }
        
        return user_roles.get(user_id, {"read"})
    
    async def _calculate_initial_trust_score(
        self, user_id: str, source_ip: str, user_agent: str
    ) -> float:
        """Calculate initial trust score for user."""
        trust_score = 0.5  # Base trust
        
        # Historical behavior bonus
        if user_id in self.behavioral_baselines:
            baseline = self.behavioral_baselines[user_id]
            if baseline.get("clean_history", False):
                trust_score += 0.3
        
        # Source IP reputation
        if source_ip in self.threat_intelligence["malicious_ips"]:
            trust_score -= 0.4
        elif source_ip.startswith("192.168.") or source_ip.startswith("10."):
            trust_score += 0.1  # Internal network bonus
        
        # User agent analysis
        if any(suspicious in user_agent.lower() 
               for suspicious in self.threat_intelligence["suspicious_user_agents"]):
            trust_score -= 0.2
        
        # Failed attempt penalty
        recent_failures = len([
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > datetime.utcnow() - timedelta(hours=1)
        ])
        trust_score -= min(0.3, recent_failures * 0.1)
        
        return max(0.0, min(1.0, trust_score))
    
    async def _identify_risk_factors(
        self, user_id: str, source_ip: str, user_agent: str, context: Dict
    ) -> List[str]:
        """Identify risk factors for the request."""
        risk_factors = []
        
        # IP-based risks
        if source_ip in self.threat_intelligence["malicious_ips"]:
            risk_factors.append("malicious_ip")
        
        # User agent risks
        if any(suspicious in user_agent.lower() 
               for suspicious in self.threat_intelligence["suspicious_user_agents"]):
            risk_factors.append("suspicious_user_agent")
        
        # Behavioral risks
        if user_id in self.suspicious_users:
            risk_factors.append("suspicious_user")
        
        # Time-based risks
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Off-hours access
            risk_factors.append("off_hours_access")
        
        # Frequency risks
        recent_requests = len([
            ts for ts in self.rate_limiters[user_id]
            if ts > time.time() - 300  # Last 5 minutes
        ])
        if recent_requests > 50:
            risk_factors.append("high_frequency_access")
        
        return risk_factors
    
    async def authorize_request(
        self, security_context: SecurityContext, 
        resource: str, 
        action: str,
        request_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Authorize request with zero-trust verification."""
        try:
            authorization_issues = []
            
            # Check basic permissions
            required_permission = f"{action}_{resource}"
            if required_permission not in security_context.permissions and "admin" not in security_context.permissions:
                authorization_issues.append(f"Missing permission: {required_permission}")
            
            # Trust score threshold check
            min_trust_threshold = self._get_trust_threshold(resource, action)
            if security_context.trust_score < min_trust_threshold:
                authorization_issues.append(f"Trust score too low: {security_context.trust_score:.2f} < {min_trust_threshold}")
            
            # Rate limiting check
            rate_limit_ok = await self._check_rate_limits(security_context, resource, action)
            if not rate_limit_ok:
                authorization_issues.append("Rate limit exceeded")
            
            # Resource-specific authorization
            resource_auth_ok = await self._check_resource_authorization(
                security_context, resource, action, request_data
            )
            if not resource_auth_ok:
                authorization_issues.append("Resource access denied")
            
            # Content security check
            content_issues = await self._check_content_security(request_data)
            authorization_issues.extend(content_issues)
            
            # Log authorization attempt
            if authorization_issues:
                await self._log_security_event(
                    SecurityEventType.AUTHORIZATION_VIOLATION,
                    f"Authorization failed: {', '.join(authorization_issues)}",
                    {"security_context": security_context, "resource": resource, "action": action}
                )
                return False, authorization_issues
            
            # Update last activity
            security_context.last_activity = datetime.utcnow()
            
            return True, []
            
        except Exception as e:
            logger.error(f"Authorization error: {e}", exc_info=True)
            return False, [f"Authorization system error: {str(e)}"]
    
    def _get_trust_threshold(self, resource: str, action: str) -> float:
        """Get required trust threshold for resource/action combination."""
        thresholds = {
            ("cost_data", "read"): 0.3,
            ("cost_data", "write"): 0.5,
            ("cost_data", "delete"): 0.7,
            ("admin_panel", "read"): 0.6,
            ("admin_panel", "write"): 0.8,
            ("user_data", "read"): 0.4,
            ("user_data", "write"): 0.6,
            ("api_keys", "read"): 0.7,
            ("api_keys", "write"): 0.9
        }
        
        return thresholds.get((resource, action), 0.5)
    
    async def _check_rate_limits(
        self, security_context: SecurityContext, resource: str, action: str
    ) -> bool:
        """Check rate limits for user and resource."""
        user_id = security_context.user_id
        current_time = time.time()
        
        # Add current request timestamp
        self.rate_limiters[user_id].append(current_time)
        
        # Check requests per minute
        recent_requests = len([
            ts for ts in self.rate_limiters[user_id]
            if ts > current_time - 60
        ])
        
        # Get rate limit from policy
        rate_policy = self.security_policies.get("rate_limiting_policy")
        if rate_policy:
            for rule in rate_policy.rules:
                if rule["type"] == "requests_per_minute":
                    if recent_requests > rule["limit"]:
                        await self._log_security_event(
                            SecurityEventType.RATE_LIMIT_VIOLATION,
                            f"Rate limit exceeded: {recent_requests} > {rule['limit']} requests/minute",
                            {"user_id": user_id, "resource": resource}
                        )
                        return False
        
        return True
    
    async def _check_resource_authorization(
        self, security_context: SecurityContext, resource: str, action: str, request_data: Dict
    ) -> bool:
        """Check resource-specific authorization rules."""
        # Cost data access control
        if resource == "cost_data":
            # Users can only access their own cost data unless admin
            if "admin" not in security_context.permissions:
                requested_user = request_data.get("user_id")
                if requested_user and requested_user != security_context.user_id:
                    return False
        
        # Admin panel access
        if resource == "admin_panel":
            return "admin" in security_context.permissions
        
        # API key management
        if resource == "api_keys":
            return "admin" in security_context.permissions or "api_management" in security_context.permissions
        
        return True
    
    async def _check_content_security(self, request_data: Dict[str, Any]) -> List[str]:
        """Check content for security violations."""
        violations = []
        
        # Check for prompt injection attempts
        prompt_text = request_data.get("prompt", "")
        if prompt_text:
            injection_detected = await self._detect_prompt_injection(prompt_text)
            if injection_detected:
                violations.append("Prompt injection detected")
        
        # Check for PII in prompts
        pii_detected = await self._detect_pii(prompt_text)
        if pii_detected:
            violations.append("PII detected in prompt")
        
        # Check for malicious patterns
        for pattern_name, pattern in self.pattern_matchers.items():
            if pattern.search(prompt_text):
                violations.append(f"Malicious pattern detected: {pattern_name}")
        
        return violations
    
    async def _detect_prompt_injection(self, prompt: str) -> bool:
        """Detect prompt injection attempts."""
        # Check against known injection patterns
        if self.pattern_matchers["prompt_injection"].search(prompt):
            return True
        
        # Check for instruction override attempts
        injection_indicators = [
            "ignore previous instructions",
            "forget your role",
            "act as a different ai",
            "you are now",
            "disregard the above",
            "new instructions:",
            "system prompt:"
        ]
        
        prompt_lower = prompt.lower()
        for indicator in injection_indicators:
            if indicator in prompt_lower:
                return True
        
        return False
    
    async def _detect_pii(self, text: str) -> bool:
        """Detect personally identifiable information."""
        # Simple PII patterns (in production, use more sophisticated detection)
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    async def detect_threats(
        self, security_context: SecurityContext, request_data: Dict[str, Any]
    ) -> List[ThreatDetection]:
        """Detect threats in real-time."""
        threats = []
        
        try:
            # Behavioral anomaly detection
            behavioral_threat = await self._detect_behavioral_anomaly(security_context, request_data)
            if behavioral_threat:
                threats.append(behavioral_threat)
            
            # Cost anomaly detection
            cost_threat = await self._detect_cost_anomaly(security_context, request_data)
            if cost_threat:
                threats.append(cost_threat)
            
            # API abuse detection
            api_abuse_threat = await self._detect_api_abuse(security_context, request_data)
            if api_abuse_threat:
                threats.append(api_abuse_threat)
            
            # Data exfiltration detection
            exfiltration_threat = await self._detect_data_exfiltration(security_context, request_data)
            if exfiltration_threat:
                threats.append(exfiltration_threat)
            
            # Log all detected threats
            for threat in threats:
                await self._log_threat_detection(threat)
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat detection error: {e}", exc_info=True)
            return []
    
    async def _detect_behavioral_anomaly(
        self, security_context: SecurityContext, request_data: Dict
    ) -> Optional[ThreatDetection]:
        """Detect behavioral anomalies."""
        user_id = security_context.user_id
        
        # Get user baseline
        baseline = self.behavioral_baselines.get(user_id, {})
        
        # Analyze current behavior
        current_behavior = {
            "request_frequency": len(self.rate_limiters[user_id]),
            "request_time": datetime.utcnow().hour,
            "source_ip": security_context.source_ip,
            "user_agent": security_context.user_agent,
            "cost_pattern": request_data.get("estimated_cost", 0)
        }
        
        # Compare with baseline
        anomaly_score = 0.0
        anomaly_indicators = []
        
        # Time-based anomaly
        normal_hours = baseline.get("normal_hours", set(range(9, 18)))
        if current_behavior["request_time"] not in normal_hours:
            anomaly_score += 0.3
            anomaly_indicators.append("unusual_time")
        
        # Frequency anomaly
        normal_frequency = baseline.get("normal_frequency", 10)
        if current_behavior["request_frequency"] > normal_frequency * 3:
            anomaly_score += 0.4
            anomaly_indicators.append("high_frequency")
        
        # Source IP anomaly
        known_ips = baseline.get("known_ips", set())
        if known_ips and current_behavior["source_ip"] not in known_ips:
            anomaly_score += 0.2
            anomaly_indicators.append("new_ip")
        
        # Create threat detection if anomaly score is high
        if anomaly_score > 0.5:
            return ThreatDetection(
                detection_id=f"behavioral_{uuid.uuid4().hex[:8]}",
                threat_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                threat_level=ThreatLevel.MEDIUM if anomaly_score < 0.8 else ThreatLevel.HIGH,
                confidence_score=anomaly_score,
                description=f"Behavioral anomaly detected for user {user_id}",
                affected_resources=[f"user:{user_id}"],
                indicators={"anomaly_score": anomaly_score, "indicators": anomaly_indicators},
                recommended_actions=["Monitor user activity", "Require additional authentication"],
                context=security_context
            )
        
        return None
    
    async def _detect_cost_anomaly(
        self, security_context: SecurityContext, request_data: Dict
    ) -> Optional[ThreatDetection]:
        """Detect cost-related anomalies and potential abuse."""
        estimated_cost = request_data.get("estimated_cost", 0)
        user_id = security_context.user_id
        
        # Get user's cost history
        user_costs = []  # In production, fetch from database
        
        # Calculate cost anomaly
        if estimated_cost > 1.0:  # High-cost request
            anomaly_indicators = ["high_cost_request"]
            threat_level = ThreatLevel.MEDIUM
            
            if estimated_cost > 10.0:  # Very high cost
                anomaly_indicators.append("very_high_cost")
                threat_level = ThreatLevel.HIGH
            
            return ThreatDetection(
                detection_id=f"cost_{uuid.uuid4().hex[:8]}",
                threat_type=SecurityEventType.ANOMALOUS_COST_PATTERN,
                threat_level=threat_level,
                confidence_score=0.8,
                description=f"High cost request detected: ${estimated_cost:.2f}",
                affected_resources=[f"user:{user_id}", "cost_budget"],
                indicators={"estimated_cost": estimated_cost, "indicators": anomaly_indicators},
                recommended_actions=["Review request necessity", "Implement cost controls"],
                context=security_context
            )
        
        return None
    
    async def _detect_api_abuse(
        self, security_context: SecurityContext, request_data: Dict
    ) -> Optional[ThreatDetection]:
        """Detect API abuse patterns."""
        user_id = security_context.user_id
        current_time = time.time()
        
        # Check for rapid-fire requests
        recent_requests = [
            ts for ts in self.rate_limiters[user_id]
            if ts > current_time - 10  # Last 10 seconds
        ]
        
        if len(recent_requests) > 20:  # More than 20 requests in 10 seconds
            return ThreatDetection(
                detection_id=f"api_abuse_{uuid.uuid4().hex[:8]}",
                threat_type=SecurityEventType.API_ABUSE,
                threat_level=ThreatLevel.HIGH,
                confidence_score=0.9,
                description=f"API abuse detected: {len(recent_requests)} requests in 10 seconds",
                affected_resources=[f"user:{user_id}", "api_endpoint"],
                indicators={"rapid_requests": len(recent_requests), "time_window": 10},
                recommended_actions=["Implement stricter rate limits", "Temporary user suspension"],
                context=security_context
            )
        
        return None
    
    async def _detect_data_exfiltration(
        self, security_context: SecurityContext, request_data: Dict
    ) -> Optional[ThreatDetection]:
        """Detect potential data exfiltration attempts."""
        prompt = request_data.get("prompt", "")
        
        # Check for data extraction patterns
        if self.pattern_matchers["data_exfiltration"].search(prompt):
            return ThreatDetection(
                detection_id=f"exfiltration_{uuid.uuid4().hex[:8]}",
                threat_type=SecurityEventType.DATA_EXFILTRATION,
                threat_level=ThreatLevel.HIGH,
                confidence_score=0.85,
                description="Potential data exfiltration attempt detected in prompt",
                affected_resources=["database", "sensitive_data"],
                indicators={"prompt_pattern": "data_extraction", "prompt_length": len(prompt)},
                recommended_actions=["Block request", "Investigate user intent", "Review data access logs"],
                context=security_context
            )
        
        return None
    
    async def _log_security_event(
        self, event_type: SecurityEventType, description: str, context: Dict
    ):
        """Log security event for audit and analysis."""
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type.value,
            "description": description,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high" if event_type in [
                SecurityEventType.AUTHENTICATION_FAILURE,
                SecurityEventType.AUTHORIZATION_VIOLATION,
                SecurityEventType.DATA_EXFILTRATION
            ] else "medium"
        }
        
        # In production, this would write to a security log aggregation system
        logger.warning(f"Security Event: {event}")
        
        # Cache for recent access
        await llm_cache.set(
            f"security_event:{event['event_id']}", 
            json.dumps(event, default=str), 
            ttl=86400
        )
    
    async def _log_threat_detection(self, threat: ThreatDetection):
        """Log detected threat."""
        self.threat_detections.append(threat)
        
        # Keep only recent detections (last 1000)
        if len(self.threat_detections) > 1000:
            self.threat_detections = self.threat_detections[-1000:]
        
        logger.warning(f"Threat Detected: {threat.threat_type.value} - {threat.description}")
    
    async def _track_failed_attempt(self, user_id: str, source_ip: str):
        """Track failed authentication attempts."""
        attempt_time = datetime.utcnow()
        self.failed_attempts[user_id].append(attempt_time)
        
        # Clean old attempts (keep last 24 hours)
        cutoff_time = attempt_time - timedelta(hours=24)
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff_time
        ]
        
        # Check for brute force attack
        recent_failures = len([
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > attempt_time - timedelta(minutes=15)
        ])
        
        if recent_failures >= 5:
            # Temporarily block IP
            self.blocked_ips.add(source_ip)
            logger.warning(f"IP {source_ip} blocked due to repeated failed attempts")
            
            # Schedule unblock (in production, use proper scheduling)
            asyncio.create_task(self._schedule_ip_unblock(source_ip, 3600))  # 1 hour
    
    async def _schedule_ip_unblock(self, ip: str, delay_seconds: int):
        """Schedule IP unblock after specified delay."""
        await asyncio.sleep(delay_seconds)
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"IP {ip} unblocked after timeout")
    
    async def _assess_request_threat(
        self, security_context: SecurityContext, request_context: Dict
    ) -> Dict[str, Any]:
        """Assess overall threat level of request."""
        threat_score = 0.0
        threat_factors = []
        
        # Risk factors contribute to threat score
        for risk_factor in security_context.risk_factors:
            if risk_factor == "malicious_ip":
                threat_score += 0.4
                threat_factors.append("malicious_ip")
            elif risk_factor == "suspicious_user":
                threat_score += 0.3
                threat_factors.append("suspicious_user")
            elif risk_factor == "off_hours_access":
                threat_score += 0.1
                threat_factors.append("off_hours_access")
            elif risk_factor == "high_frequency_access":
                threat_score += 0.2
                threat_factors.append("high_frequency")
        
        return {
            "threat_score": min(1.0, threat_score),
            "threat_factors": threat_factors,
            "assessment_time": datetime.utcnow()
        }
    
    async def _calculate_trust_score(
        self, security_context: SecurityContext, threat_assessment: Dict
    ) -> float:
        """Calculate updated trust score based on threat assessment."""
        base_trust = security_context.trust_score
        threat_score = threat_assessment.get("threat_score", 0)
        
        # Adjust trust based on threat level
        adjusted_trust = base_trust - (threat_score * 0.5)
        
        # Time-based trust decay
        time_since_creation = (datetime.utcnow() - security_context.created_at).total_seconds()
        if time_since_creation > 3600:  # 1 hour
            time_decay = min(0.2, time_since_creation / 18000)  # Max 0.2 decay over 5 hours
            adjusted_trust -= time_decay
        
        return max(0.0, min(1.0, adjusted_trust))
    
    async def _continuous_threat_monitoring(self):
        """Continuously monitor for threats."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Monitor for suspicious patterns
                await self._analyze_global_threat_patterns()
                
                # Update behavioral baselines
                await self._update_behavioral_baselines()
                
                # Clean up old data
                await self._cleanup_security_data()
                
            except Exception as e:
                logger.error(f"Continuous threat monitoring error: {e}")
    
    async def _analyze_global_threat_patterns(self):
        """Analyze global threat patterns across all users."""
        # Analyze recent threat detections for patterns
        if len(self.threat_detections) > 10:
            recent_threats = self.threat_detections[-50:]
            
            # Group by threat type
            threat_counts = defaultdict(int)
            for threat in recent_threats:
                threat_counts[threat.threat_type] += 1
            
            # Check for attack waves
            for threat_type, count in threat_counts.items():
                if count > 10:  # More than 10 of same threat type
                    logger.warning(f"Potential attack wave detected: {threat_type.value} ({count} instances)")
    
    async def _update_behavioral_baselines(self):
        """Update behavioral baselines for users."""
        for user_id, context in self.security_contexts.items():
            if user_id not in self.behavioral_baselines:
                self.behavioral_baselines[user_id] = {
                    "normal_hours": set(),
                    "known_ips": set(),
                    "normal_frequency": 0,
                    "clean_history": True
                }
            
            baseline = self.behavioral_baselines[user_id]
            
            # Update normal hours
            current_hour = datetime.utcnow().hour
            baseline["normal_hours"].add(current_hour)
            
            # Update known IPs
            baseline["known_ips"].add(context.source_ip)
            
            # Update frequency baseline
            recent_requests = len([
                ts for ts in self.rate_limiters[user_id]
                if ts > time.time() - 3600  # Last hour
            ])
            if baseline["normal_frequency"] == 0:
                baseline["normal_frequency"] = recent_requests
            else:
                # Exponential moving average
                baseline["normal_frequency"] = baseline["normal_frequency"] * 0.9 + recent_requests * 0.1
    
    async def _cleanup_security_data(self):
        """Clean up old security data."""
        current_time = datetime.utcnow()
        
        # Clean old security contexts (older than 8 hours)
        expired_sessions = [
            session_id for session_id, context in self.security_contexts.items()
            if current_time - context.last_activity > timedelta(hours=8)
        ]
        
        for session_id in expired_sessions:
            del self.security_contexts[session_id]
        
        # Clean old failed attempts
        for user_id in list(self.failed_attempts.keys()):
            cutoff_time = current_time - timedelta(hours=24)
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > cutoff_time
            ]
            
            if not self.failed_attempts[user_id]:
                del self.failed_attempts[user_id]
    
    async def _behavioral_analysis_engine(self):
        """Advanced behavioral analysis engine."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze user behavior patterns
                for user_id in list(self.security_contexts.keys()):
                    await self._analyze_user_behavior(user_id)
                
            except Exception as e:
                logger.error(f"Behavioral analysis error: {e}")
    
    async def _analyze_user_behavior(self, user_id: str):
        """Analyze specific user behavior."""
        # Get user's recent activity
        user_requests = [
            ts for ts in self.rate_limiters[user_id]
            if ts > time.time() - 3600  # Last hour
        ]
        
        if len(user_requests) > 50:  # High activity
            # Check for suspicious patterns
            await self._check_suspicious_patterns(user_id, user_requests)
    
    async def _check_suspicious_patterns(self, user_id: str, request_times: List[float]):
        """Check for suspicious behavior patterns."""
        # Check for bot-like regular intervals
        if len(request_times) > 10:
            intervals = [request_times[i] - request_times[i-1] for i in range(1, len(request_times))]
            avg_interval = sum(intervals) / len(intervals)
            
            # If requests are too regular (bot-like behavior)
            if all(abs(interval - avg_interval) < 0.5 for interval in intervals):
                if user_id not in self.suspicious_users:
                    self.suspicious_users.add(user_id)
                    logger.warning(f"Bot-like behavior detected for user {user_id}")
    
    async def _security_policy_enforcement(self):
        """Enforce security policies."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Enforce session timeouts
                await self._enforce_session_timeouts()
                
                # Enforce concurrent session limits
                await self._enforce_concurrent_session_limits()
                
            except Exception as e:
                logger.error(f"Security policy enforcement error: {e}")
    
    async def _enforce_session_timeouts(self):
        """Enforce session timeout policies."""
        timeout_policy = self.security_policies.get("auth_policy")
        if not timeout_policy:
            return
        
        timeout_minutes = 60  # Default
        for rule in timeout_policy.rules:
            if rule["type"] == "session_timeout":
                timeout_minutes = rule["minutes"]
                break
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        expired_sessions = [
            session_id for session_id, context in self.security_contexts.items()
            if context.last_activity < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.security_contexts[session_id]
            logger.info(f"Session {session_id} expired due to timeout")
    
    async def _enforce_concurrent_session_limits(self):
        """Enforce concurrent session limits."""
        auth_policy = self.security_policies.get("auth_policy")
        if not auth_policy:
            return
        
        max_sessions = 5  # Default
        for rule in auth_policy.rules:
            if rule["type"] == "concurrent_sessions":
                max_sessions = rule["max_sessions"]
                break
        
        # Group sessions by user
        user_sessions = defaultdict(list)
        for session_id, context in self.security_contexts.items():
            user_sessions[context.user_id].append((session_id, context))
        
        # Enforce limits
        for user_id, sessions in user_sessions.items():
            if len(sessions) > max_sessions:
                # Sort by last activity and remove oldest
                sessions.sort(key=lambda x: x[1].last_activity)
                excess_sessions = sessions[:-max_sessions]
                
                for session_id, _ in excess_sessions:
                    del self.security_contexts[session_id]
                    logger.info(f"Session {session_id} terminated due to concurrent session limit")
    
    async def _session_management(self):
        """Manage user sessions."""
        while True:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes
                
                # Update trust scores for active sessions
                for context in self.security_contexts.values():
                    # Gradual trust score improvement for good behavior
                    if len(context.risk_factors) == 0:
                        context.trust_score = min(1.0, context.trust_score + 0.01)
                
            except Exception as e:
                logger.error(f"Session management error: {e}")
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security engine status."""
        current_time = datetime.utcnow()
        
        # Calculate threat statistics
        recent_threats = [
            threat for threat in self.threat_detections
            if current_time - threat.detected_at < timedelta(hours=24)
        ]
        
        threat_by_level = defaultdict(int)
        threat_by_type = defaultdict(int)
        
        for threat in recent_threats:
            threat_by_level[threat.threat_level.value] += 1
            threat_by_type[threat.threat_type.value] += 1
        
        return {
            "status": "active",
            "active_sessions": len(self.security_contexts),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_users": len(self.suspicious_users),
            "security_policies": len(self.security_policies),
            "threat_detections_24h": len(recent_threats),
            "threats_by_level": dict(threat_by_level),
            "threats_by_type": dict(threat_by_type),
            "failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
            "behavioral_baselines": len(self.behavioral_baselines),
            "average_trust_score": np.mean([
                context.trust_score for context in self.security_contexts.values()
            ]) if self.security_contexts else 0.0,
            "last_threat_detection": max([
                threat.detected_at for threat in recent_threats
            ]).isoformat() if recent_threats else None
        }


# Global zero-trust security engine instance
zero_trust_security_engine = ZeroTrustSecurityEngine()