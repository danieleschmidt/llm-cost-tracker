"""Enhanced security module for sentiment analysis with threat detection and prevention."""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from datetime import datetime, timedelta

from .validation import ValidationError, SecurityValidationError
from .quantum_i18n import t

logger = logging.getLogger(__name__)

class ThreatLevel(str, Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    """Types of security threats."""
    INJECTION_ATTEMPT = "injection_attempt"
    PII_EXPOSURE = "pii_exposure"
    MALICIOUS_CONTENT = "malicious_content"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    CONTENT_MANIPULATION = "content_manipulation"
    DATA_EXFILTRATION = "data_exfiltration"

@dataclass
class SecurityThreat:
    """Security threat detection result."""
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    detected_patterns: List[str]
    recommended_action: str
    risk_score: float  # 0.0 to 10.0
    mitigation_applied: bool = False

class SentimentSecurityScanner:
    """Comprehensive security scanner for sentiment analysis inputs."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.pii_patterns = self._load_pii_patterns()
        self.malicious_keywords = self._load_malicious_keywords()
        self.content_filters = self._load_content_filters()
        
        # Security metrics
        self.scan_history: List[Dict] = []
        self.threat_counts: Dict[ThreatType, int] = {threat: 0 for threat in ThreatType}
        self.blocked_requests = 0
        self.total_scans = 0
        
        logger.info("SentimentSecurityScanner initialized with comprehensive threat detection")
    
    def _load_threat_patterns(self) -> Dict[ThreatType, List[re.Pattern]]:
        """Load compiled regex patterns for threat detection."""
        patterns = {
            ThreatType.INJECTION_ATTEMPT: [
                re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
                re.compile(r'javascript:', re.IGNORECASE),
                re.compile(r'on\w+\s*=', re.IGNORECASE),
                re.compile(r'eval\s*\(', re.IGNORECASE),
                re.compile(r'document\.write', re.IGNORECASE),
                re.compile(r'window\.location', re.IGNORECASE),
                re.compile(r'exec\s*\(', re.IGNORECASE),
                re.compile(r'system\s*\(', re.IGNORECASE),
                re.compile(r'[\;\|\&].*?(rm|del|format)', re.IGNORECASE),
                re.compile(r'(union|select|insert|update|delete|drop)\s+.*?(from|into|table)', re.IGNORECASE),
            ],
            
            ThreatType.CONTENT_MANIPULATION: [
                re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'),  # Control characters
                re.compile(r'\u200B|\u200C|\u200D|\uFEFF'),  # Zero-width characters
                re.compile(r'(.)\1{50,}'),  # Excessive character repetition
                re.compile(r'[^\x20-\x7E\s]{100,}'),  # Long sequences of non-printable chars
            ],
            
            ThreatType.DATA_EXFILTRATION: [
                re.compile(r'curl\s+.*?http', re.IGNORECASE),
                re.compile(r'wget\s+.*?http', re.IGNORECASE),
                re.compile(r'fetch\s*\(.*?http', re.IGNORECASE),
                re.compile(r'XMLHttpRequest', re.IGNORECASE),
                re.compile(r'base64|atob|btoa', re.IGNORECASE),
            ]
        }
        
        return patterns
    
    def _load_pii_patterns(self) -> List[re.Pattern]:
        """Load patterns for PII detection."""
        return [
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Phone numbers (various formats)
            re.compile(r'\b(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            re.compile(r'\b\+?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b'),
            
            # Social Security Numbers
            re.compile(r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'),
            re.compile(r'\b[0-9]{9}\b'),
            
            # Credit card numbers
            re.compile(r'\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b'),
            
            # IP addresses
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            
            # URLs with potential sensitive info
            re.compile(r'https?://[^\s]+(?:token|key|password|secret)[^\s]*', re.IGNORECASE),
            
            # Common sensitive keywords with numbers/special chars (likely credentials)
            re.compile(r'\b(?:password|passwd|pwd|secret|token|key|api_key)\s*[:=]\s*[^\s]+', re.IGNORECASE),
        ]
    
    def _load_malicious_keywords(self) -> Set[str]:
        """Load malicious keywords for content filtering."""
        return {
            # Command injection keywords
            'eval', 'exec', 'system', 'shell_exec', 'passthru', 'proc_open',
            'popen', 'curl_exec', 'curl_multi_exec', 'parse_ini_file',
            'show_source', 'file_get_contents', 'readfile', 'highlight_file',
            
            # Script injection
            'javascript', 'vbscript', 'onload', 'onerror', 'onclick',
            'onmouseover', 'onfocus', 'onblur', 'onchange', 'onsubmit',
            
            # SQL injection
            'union', 'select', 'insert', 'update', 'delete', 'drop',
            'create', 'alter', 'truncate', 'replace', 'handler',
            
            # System commands
            'chmod', 'chown', 'rm', 'del', 'format', 'fdisk',
            'kill', 'killall', 'shutdown', 'reboot', 'halt',
        }
    
    def _load_content_filters(self) -> Dict[str, re.Pattern]:
        """Load content filtering patterns."""
        return {
            'excessive_caps': re.compile(r'[A-Z]{20,}'),
            'suspicious_urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'encoded_content': re.compile(r'%[0-9a-fA-F]{2}|&#x?[0-9a-fA-F]+;|\\u[0-9a-fA-F]{4}'),
            'suspicious_unicode': re.compile(r'[\u200B-\u200F\u202A-\u202E\u2060-\u206F]'),
        }
    
    async def scan_text(
        self, 
        text: str, 
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Tuple[bool, List[SecurityThreat]]:
        """
        Perform comprehensive security scan on input text.
        
        Returns:
            Tuple of (is_safe: bool, threats: List[SecurityThreat])
        """
        if not text or not isinstance(text, str):
            return True, []
        
        scan_start = time.perf_counter()
        threats: List[SecurityThreat] = []
        
        self.total_scans += 1
        
        try:
            # 1. Check for injection attempts
            injection_threats = await self._scan_injection_attempts(text)
            threats.extend(injection_threats)
            
            # 2. Check for PII exposure
            pii_threats = await self._scan_pii_exposure(text)
            threats.extend(pii_threats)
            
            # 3. Check for malicious content
            malicious_threats = await self._scan_malicious_content(text)
            threats.extend(malicious_threats)
            
            # 4. Check for content manipulation
            manipulation_threats = await self._scan_content_manipulation(text)
            threats.extend(manipulation_threats)
            
            # 5. Check for data exfiltration attempts
            exfiltration_threats = await self._scan_data_exfiltration(text)
            threats.extend(exfiltration_threats)
            
            # Calculate overall risk assessment
            is_safe = self._assess_overall_risk(threats)
            
            # Update metrics
            for threat in threats:
                self.threat_counts[threat.threat_type] += 1
            
            if not is_safe:
                self.blocked_requests += 1
            
            # Log scan results
            scan_time = (time.perf_counter() - scan_start) * 1000
            self._log_scan_result(text, threats, is_safe, scan_time, user_id)
            
            return is_safe, threats
            
        except Exception as e:
            logger.error(f"Security scan failed: {str(e)}", exc_info=True)
            # Fail secure - block on scan errors
            return False, [
                SecurityThreat(
                    threat_type=ThreatType.MALICIOUS_CONTENT,
                    threat_level=ThreatLevel.HIGH,
                    description="Security scan failure - blocking for safety",
                    detected_patterns=[],
                    recommended_action="Block request",
                    risk_score=8.0
                )
            ]
    
    async def _scan_injection_attempts(self, text: str) -> List[SecurityThreat]:
        """Scan for code injection attempts."""
        threats = []
        
        for pattern in self.threat_patterns[ThreatType.INJECTION_ATTEMPT]:
            matches = pattern.findall(text)
            if matches:
                threat = SecurityThreat(
                    threat_type=ThreatType.INJECTION_ATTEMPT,
                    threat_level=ThreatLevel.CRITICAL,
                    description=f"Code injection attempt detected: {len(matches)} patterns found",
                    detected_patterns=[str(match)[:100] for match in matches[:5]],  # Limit for logging
                    recommended_action="Block request immediately",
                    risk_score=9.5
                )
                threats.append(threat)
                break  # One critical threat is enough
        
        return threats
    
    async def _scan_pii_exposure(self, text: str) -> List[SecurityThreat]:
        """Scan for personally identifiable information."""
        threats = []
        detected_pii = []
        
        for pattern in self.pii_patterns:
            matches = pattern.findall(text)
            if matches:
                detected_pii.extend([str(match) for match in matches[:3]])  # Limit for privacy
        
        if detected_pii:
            threat = SecurityThreat(
                threat_type=ThreatType.PII_EXPOSURE,
                threat_level=ThreatLevel.HIGH,
                description=f"PII detected in input: {len(detected_pii)} instances",
                detected_patterns=["[REDACTED]"] * len(detected_pii),  # Don't log actual PII
                recommended_action="Anonymize or reject request",
                risk_score=7.5
            )
            threats.append(threat)
        
        return threats
    
    async def _scan_malicious_content(self, text: str) -> List[SecurityThreat]:
        """Scan for malicious keywords and patterns."""
        threats = []
        text_lower = text.lower()
        detected_keywords = []
        
        for keyword in self.malicious_keywords:
            if keyword in text_lower:
                detected_keywords.append(keyword)
        
        if detected_keywords:
            risk_score = min(8.0, 2.0 + len(detected_keywords) * 0.5)  # Scale with number of keywords
            threat_level = ThreatLevel.HIGH if risk_score > 6.0 else ThreatLevel.MEDIUM
            
            threat = SecurityThreat(
                threat_type=ThreatType.MALICIOUS_CONTENT,
                threat_level=threat_level,
                description=f"Malicious keywords detected: {len(detected_keywords)} found",
                detected_patterns=detected_keywords[:10],  # Limit for logging
                recommended_action="Review and potentially block",
                risk_score=risk_score
            )
            threats.append(threat)
        
        return threats
    
    async def _scan_content_manipulation(self, text: str) -> List[SecurityThreat]:
        """Scan for content manipulation attempts."""
        threats = []
        
        for filter_name, pattern in self.content_filters.items():
            matches = pattern.findall(text)
            if matches:
                risk_score = 4.0 + len(matches) * 0.2
                threat_level = ThreatLevel.MEDIUM if risk_score > 5.0 else ThreatLevel.LOW
                
                threat = SecurityThreat(
                    threat_type=ThreatType.CONTENT_MANIPULATION,
                    threat_level=threat_level,
                    description=f"Content manipulation detected: {filter_name}",
                    detected_patterns=[filter_name],
                    recommended_action="Sanitize content" if threat_level == ThreatLevel.LOW else "Block request",
                    risk_score=risk_score
                )
                threats.append(threat)
        
        return threats
    
    async def _scan_data_exfiltration(self, text: str) -> List[SecurityThreat]:
        """Scan for data exfiltration attempts."""
        threats = []
        
        for pattern in self.threat_patterns[ThreatType.DATA_EXFILTRATION]:
            matches = pattern.findall(text)
            if matches:
                threat = SecurityThreat(
                    threat_type=ThreatType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Data exfiltration attempt detected: {len(matches)} patterns",
                    detected_patterns=[str(match)[:50] for match in matches[:3]],
                    recommended_action="Block request and investigate",
                    risk_score=8.5
                )
                threats.append(threat)
                break  # One detection is enough
        
        return threats
    
    def _assess_overall_risk(self, threats: List[SecurityThreat]) -> bool:
        """Assess overall risk and determine if request should be allowed."""
        if not threats:
            return True
        
        # Calculate composite risk score
        total_risk = sum(threat.risk_score for threat in threats)
        max_risk = max(threat.risk_score for threat in threats)
        
        # Critical threats are always blocked
        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
        if critical_threats:
            return False
        
        # High risk threshold
        high_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]
        if len(high_threats) >= 2 or max_risk >= 8.0:
            return False
        
        # Medium risk threshold
        medium_threats = [t for t in threats if t.threat_level == ThreatLevel.MEDIUM]
        if len(medium_threats) >= 3 or total_risk >= 15.0:
            return False
        
        # Allow low risk requests
        return True
    
    def _log_scan_result(
        self, 
        text: str, 
        threats: List[SecurityThreat], 
        is_safe: bool, 
        scan_time_ms: float,
        user_id: Optional[str] = None
    ):
        """Log security scan results for audit trail."""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        log_data = {
            "text_hash": text_hash,
            "text_length": len(text),
            "threats_detected": len(threats),
            "is_safe": is_safe,
            "scan_time_ms": scan_time_ms,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if threats:
            log_data["threat_types"] = [t.threat_type.value for t in threats]
            log_data["max_risk_score"] = max(t.risk_score for t in threats)
            log_data["threat_levels"] = [t.threat_level.value for t in threats]
        
        # Store in scan history (keep last 1000 scans)
        self.scan_history.append(log_data)
        if len(self.scan_history) > 1000:
            self.scan_history.pop(0)
        
        if not is_safe:
            logger.warning(f"SECURITY: Blocked malicious request", extra=log_data)
        elif threats:
            logger.info(f"SECURITY: Threats detected but allowed", extra=log_data)
        else:
            logger.debug(f"SECURITY: Clean request processed", extra=log_data)
    
    def get_security_metrics(self) -> Dict:
        """Get security scanning metrics."""
        return {
            "total_scans": self.total_scans,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(1, self.total_scans),
            "threat_counts": {k.value: v for k, v in self.threat_counts.items()},
            "recent_scan_count": len(self.scan_history),
            "avg_scan_time_ms": (
                sum(scan.get("scan_time_ms", 0) for scan in self.scan_history[-100:]) / 
                min(100, len(self.scan_history))
            ) if self.scan_history else 0
        }
    
    async def scan_batch(self, texts: List[str], user_id: Optional[str] = None) -> Tuple[List[bool], List[List[SecurityThreat]]]:
        """Scan multiple texts in batch with optimized processing."""
        if not texts:
            return [], []
        
        # Process in parallel for better performance
        tasks = [self.scan_text(text, user_id) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        is_safe_list = []
        threats_list = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch security scan failed for item {i}: {str(result)}")
                # Fail secure
                is_safe_list.append(False)
                threats_list.append([
                    SecurityThreat(
                        threat_type=ThreatType.MALICIOUS_CONTENT,
                        threat_level=ThreatLevel.HIGH,
                        description="Batch scan failure",
                        detected_patterns=[],
                        recommended_action="Block request",
                        risk_score=8.0
                    )
                ])
            else:
                is_safe, threats = result
                is_safe_list.append(is_safe)
                threats_list.append(threats)
        
        return is_safe_list, threats_list
    
    def cleanup_old_history(self, hours: int = 24):
        """Clean up old scan history to manage memory."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        self.scan_history = [
            scan for scan in self.scan_history 
            if scan.get("timestamp", "") >= cutoff_iso
        ]

# Global security scanner instance
sentiment_security_scanner = SentimentSecurityScanner()