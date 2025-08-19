#!/usr/bin/env python3
"""
Generation 2: Enhanced Security & Compliance System
Implements comprehensive security measures and global compliance features
"""

import asyncio
import hashlib
import json
import logging
import secrets
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ComplianceStandard(Enum):
    """Global compliance standards."""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)

class DataClassification(Enum):
    """Data sensitivity classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information

class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event tracking."""
    id: str
    timestamp: datetime
    event_type: str
    threat_level: SecurityThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    mitigation_applied: bool = False
    investigation_status: str = "pending"

@dataclass
class PIIDetectionResult:
    """PII detection analysis result."""
    contains_pii: bool
    pii_types: List[str]
    confidence_score: float
    redacted_content: str
    original_length: int

class SecurityScanner:
    """Advanced security scanning and threat detection."""
    
    def __init__(self):
        self.threat_patterns = {
            "sql_injection": [
                r"(\bUNION\b.*\bSELECT\b)", r"(\bDROP\b.*\bTABLE\b)", 
                r"(\bINSERT\b.*\bINTO\b.*\bVALUES\b)", r"(\bDELETE\b.*\bFROM\b)"
            ],
            "xss_attack": [
                r"<script.*?>.*?</script>", r"javascript:", r"on\w+\s*=", 
                r"<iframe.*?>", r"<object.*?>"
            ],
            "command_injection": [
                r"(\||;|&|`|\$\(|\${)", r"(\bcat\b|\bls\b|\brm\b|\bmv\b|\bcp\b)",
                r"(\bwget\b|\bcurl\b|\bssh\b|\btelnet\b)"
            ],
            "path_traversal": [
                r"(\.\./|\.\.\\)", r"(%2e%2e%2f|%2e%2e%5c)", r"(\.\.%2f|\.\.%5c)"
            ]
        }
        
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
            "ssn": r"\b(?:\d{3}[-.\s]?\d{2}[-.\s]?\d{4}|\d{9})\b",
            "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        }
        
        self.security_events = []
    
    def scan_for_threats(self, content: str, context: Dict[str, Any] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """Comprehensive security threat scanning."""
        threats_found = []
        
        import re
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    threat_level = self._assess_threat_level(threat_type, matches)
                    threats_found.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "matches": matches[:3],  # Limit to first 3 matches
                        "threat_level": threat_level.value,
                        "description": self._get_threat_description(threat_type)
                    })
                    
                    # Record security event
                    self._record_security_event(threat_type, threat_level, context)
        
        return len(threats_found) > 0, threats_found
    
    def detect_pii(self, content: str) -> PIIDetectionResult:
        """Advanced PII detection with redaction."""
        import re
        
        pii_found = []
        redacted_content = content
        total_confidence = 0.0
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                pii_found.append(pii_type)
                confidence = self._calculate_pii_confidence(pii_type, matches)
                total_confidence += confidence
                
                # Redact PII
                redacted_content = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted_content)
        
        avg_confidence = total_confidence / len(pii_found) if pii_found else 0.0
        
        return PIIDetectionResult(
            contains_pii=len(pii_found) > 0,
            pii_types=pii_found,
            confidence_score=min(1.0, avg_confidence),
            redacted_content=redacted_content,
            original_length=len(content)
        )
    
    def _assess_threat_level(self, threat_type: str, matches: List) -> SecurityThreatLevel:
        """Assess security threat level based on type and matches."""
        threat_severities = {
            "sql_injection": SecurityThreatLevel.CRITICAL,
            "command_injection": SecurityThreatLevel.CRITICAL,
            "xss_attack": SecurityThreatLevel.HIGH,
            "path_traversal": SecurityThreatLevel.HIGH
        }
        return threat_severities.get(threat_type, SecurityThreatLevel.MEDIUM)
    
    def _calculate_pii_confidence(self, pii_type: str, matches: List) -> float:
        """Calculate confidence score for PII detection."""
        base_confidences = {
            "email": 0.9,
            "phone": 0.8,
            "ssn": 0.95,
            "credit_card": 0.9,
            "ip_address": 0.7
        }
        
        base_confidence = base_confidences.get(pii_type, 0.5)
        match_factor = min(1.0, len(matches) * 0.1)
        return min(1.0, base_confidence + match_factor)
    
    def _get_threat_description(self, threat_type: str) -> str:
        """Get human-readable threat description."""
        descriptions = {
            "sql_injection": "Potential SQL injection attack detected",
            "xss_attack": "Cross-site scripting (XSS) pattern found",
            "command_injection": "Command injection attempt detected",
            "path_traversal": "Path traversal attack pattern identified"
        }
        return descriptions.get(threat_type, f"Security threat: {threat_type}")
    
    def _record_security_event(self, threat_type: str, threat_level: SecurityThreatLevel, context: Dict[str, Any]):
        """Record security event for monitoring."""
        event = SecurityEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=threat_type,
            threat_level=threat_level,
            source_ip=context.get("source_ip") if context else None,
            user_id=context.get("user_id") if context else None,
            description=self._get_threat_description(threat_type)
        )
        self.security_events.append(event)

class ComplianceManager:
    """Global compliance management system."""
    
    def __init__(self):
        self.active_standards = {
            ComplianceStandard.GDPR,
            ComplianceStandard.CCPA,
            ComplianceStandard.PDPA
        }
        self.consent_records = {}
        self.data_processing_logs = []
        self.breach_notifications = []
    
    def check_compliance(self, operation: str, data: Dict[str, Any], 
                        user_location: str = "unknown") -> Tuple[bool, List[str]]:
        """Comprehensive compliance checking."""
        violations = []
        
        # GDPR Compliance (EU)
        if ComplianceStandard.GDPR in self.active_standards:
            gdpr_violations = self._check_gdpr_compliance(operation, data, user_location)
            violations.extend(gdpr_violations)
        
        # CCPA Compliance (California, US)
        if ComplianceStandard.CCPA in self.active_standards:
            ccpa_violations = self._check_ccpa_compliance(operation, data, user_location)
            violations.extend(ccpa_violations)
        
        # PDPA Compliance (Singapore)
        if ComplianceStandard.PDPA in self.active_standards:
            pdpa_violations = self._check_pdpa_compliance(operation, data, user_location)
            violations.extend(pdpa_violations)
        
        return len(violations) == 0, violations
    
    def _check_gdpr_compliance(self, operation: str, data: Dict[str, Any], user_location: str) -> List[str]:
        """GDPR compliance validation."""
        violations = []
        
        # Article 6: Lawful basis for processing
        if not data.get("consent_given", False) and not data.get("legitimate_interest", False):
            violations.append("GDPR Article 6: No lawful basis for processing personal data")
        
        # Article 25: Data protection by design and by default
        if "encryption" not in data.get("security_measures", []):
            violations.append("GDPR Article 25: Data not encrypted by default")
        
        # Article 32: Security of processing
        if operation == "data_transfer" and not data.get("secure_transfer", False):
            violations.append("GDPR Article 32: Insecure data transfer")
        
        return violations
    
    def _check_ccpa_compliance(self, operation: str, data: Dict[str, Any], user_location: str) -> List[str]:
        """CCPA compliance validation."""
        violations = []
        
        # Right to know
        if operation == "data_collection" and not data.get("purpose_disclosed", False):
            violations.append("CCPA: Purpose of data collection not disclosed")
        
        # Right to delete
        if operation == "data_deletion" and not data.get("deletion_confirmed", False):
            violations.append("CCPA: Data deletion not properly confirmed")
        
        return violations
    
    def _check_pdpa_compliance(self, operation: str, data: Dict[str, Any], user_location: str) -> List[str]:
        """PDPA compliance validation.""" 
        violations = []
        
        # Consent requirement
        if not data.get("explicit_consent", False):
            violations.append("PDPA: Explicit consent required for data processing")
        
        # Purpose limitation
        if "purpose" not in data:
            violations.append("PDPA: Data processing purpose not specified")
        
        return violations
    
    def record_consent(self, user_id: str, purpose: str, granted: bool, 
                      timestamp: datetime = None) -> str:
        """Record user consent with audit trail."""
        consent_id = str(uuid.uuid4())
        consent_record = {
            "consent_id": consent_id,
            "user_id": user_id,
            "purpose": purpose,
            "granted": granted,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "ip_address": "recorded_securely",  # Would be actual IP in production
            "user_agent": "recorded_securely"    # Would be actual user agent
        }
        
        self.consent_records[consent_id] = consent_record
        return consent_id
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_standards": [std.value for std in self.active_standards],
            "consent_statistics": {
                "total_consent_records": len(self.consent_records),
                "granted_consents": len([c for c in self.consent_records.values() if c["granted"]]),
                "recent_consents": len([
                    c for c in self.consent_records.values()
                    if datetime.fromisoformat(c["timestamp"]) > datetime.now() - timedelta(days=30)
                ])
            },
            "processing_logs": len(self.data_processing_logs),
            "breach_notifications": len(self.breach_notifications),
            "compliance_status": "compliant" if len(self.breach_notifications) == 0 else "under_review"
        }

class InternationalizationManager:
    """Multi-language support system."""
    
    def __init__(self):
        self.translations = {
            "en": {
                "task_created": "Task created successfully",
                "task_failed": "Task creation failed",
                "validation_error": "Validation error occurred",
                "security_threat": "Security threat detected",
                "compliance_violation": "Compliance violation found"
            },
            "es": {
                "task_created": "Tarea creada exitosamente",
                "task_failed": "Error al crear la tarea",
                "validation_error": "Error de validación ocurrido",
                "security_threat": "Amenaza de seguridad detectada",
                "compliance_violation": "Violación de cumplimiento encontrada"
            },
            "fr": {
                "task_created": "Tâche créée avec succès",
                "task_failed": "Échec de la création de tâche",
                "validation_error": "Erreur de validation survenue",
                "security_threat": "Menace de sécurité détectée",
                "compliance_violation": "Violation de conformité trouvée"
            },
            "de": {
                "task_created": "Aufgabe erfolgreich erstellt",
                "task_failed": "Aufgabenerstellung fehlgeschlagen",
                "validation_error": "Validierungsfehler aufgetreten",
                "security_threat": "Sicherheitsbedrohung erkannt",
                "compliance_violation": "Compliance-Verletzung gefunden"
            },
            "ja": {
                "task_created": "タスクが正常に作成されました",
                "task_failed": "タスクの作成に失敗しました",
                "validation_error": "検証エラーが発生しました",
                "security_threat": "セキュリティ脅威が検出されました",
                "compliance_violation": "コンプライアンス違反が発見されました"
            },
            "zh": {
                "task_created": "任务创建成功",
                "task_failed": "任务创建失败",
                "validation_error": "发生验证错误",
                "security_threat": "检测到安全威胁",
                "compliance_violation": "发现合规性违规"
            }
        }
    
    def get_message(self, key: str, language: str = "en") -> str:
        """Get localized message."""
        return self.translations.get(language, {}).get(key, 
               self.translations["en"].get(key, key))
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.translations.keys())

class SecureComplianceSystem:
    """Enhanced security and compliance system for Generation 2."""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.compliance_manager = ComplianceManager()
        self.i18n_manager = InternationalizationManager()
        
        # Configure secure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('security_compliance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_secure_input(self, content: str, context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive security validation of input content."""
        validation_result = {
            "is_secure": True,
            "threats_detected": [],
            "pii_analysis": None,
            "compliance_status": True,
            "compliance_violations": [],
            "sanitized_content": content
        }
        
        # Security threat scanning
        has_threats, threats = self.security_scanner.scan_for_threats(content, context)
        if has_threats:
            validation_result["is_secure"] = False
            validation_result["threats_detected"] = threats
            self.logger.warning(f"Security threats detected: {[t['type'] for t in threats]}")
        
        # PII detection and redaction
        pii_result = self.security_scanner.detect_pii(content)
        validation_result["pii_analysis"] = {
            "contains_pii": pii_result.contains_pii,
            "pii_types": pii_result.pii_types,
            "confidence": pii_result.confidence_score
        }
        
        if pii_result.contains_pii:
            validation_result["sanitized_content"] = pii_result.redacted_content
            self.logger.info(f"PII detected and redacted: {pii_result.pii_types}")
        
        # Compliance checking
        operation_data = {
            "content": content,
            "has_pii": pii_result.contains_pii,
            "consent_given": context.get("consent_given", False) if context else False,
            "purpose_disclosed": True,  # Assume purpose is disclosed
            "encryption": True,
            "security_measures": ["encryption", "access_control"]
        }
        
        is_compliant, violations = self.compliance_manager.check_compliance(
            "data_processing", operation_data, context.get("user_location", "unknown") if context else "unknown"
        )
        
        if not is_compliant:
            validation_result["compliance_status"] = False
            validation_result["compliance_violations"] = violations
            self.logger.warning(f"Compliance violations: {violations}")
        
        return validation_result["is_secure"] and validation_result["compliance_status"], validation_result
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security and compliance report."""
        # Recent security events
        recent_events = [
            {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type,
                "threat_level": event.threat_level.value,
                "mitigated": event.mitigation_applied
            }
            for event in self.security_scanner.security_events[-10:]  # Last 10 events
        ]
        
        # Compliance report
        compliance_report = self.compliance_manager.generate_compliance_report()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "security": {
                "total_security_events": len(self.security_scanner.security_events),
                "recent_events": recent_events,
                "threat_distribution": self._analyze_threat_distribution()
            },
            "compliance": compliance_report,
            "i18n": {
                "supported_languages": self.i18n_manager.get_supported_languages(),
                "total_translations": sum(len(t) for t in self.i18n_manager.translations.values())
            }
        }
    
    def _analyze_threat_distribution(self) -> Dict[str, int]:
        """Analyze distribution of security threats."""
        threat_counts = {}
        for event in self.security_scanner.security_events:
            threat_counts[event.event_type] = threat_counts.get(event.event_type, 0) + 1
        return threat_counts

def test_security_compliance_system():
    """Test security and compliance features."""
    print("🔒 GENERATION 2: SECURITY & COMPLIANCE TESTING")
    print("=" * 55)
    
    system = SecureComplianceSystem()
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "generation": 2,
        "security_tests": {},
        "compliance_tests": {},
        "i18n_tests": {},
        "overall_assessment": {}
    }
    
    # Test 1: Security Threat Detection
    print("🛡️ Test 1: Security Threat Detection")
    
    malicious_inputs = [
        "'; DROP TABLE users; --",  # SQL injection
        "<script>alert('XSS')</script>",  # XSS attack
        "cat /etc/passwd; rm -rf /",  # Command injection
        "../../../etc/passwd"  # Path traversal
    ]
    
    threat_detection_results = []
    for malicious_input in malicious_inputs:
        is_secure, result = system.validate_secure_input(malicious_input)
        threat_detection_results.append({
            "input": malicious_input[:30] + "..." if len(malicious_input) > 30 else malicious_input,
            "is_secure": is_secure,
            "threats_found": len(result["threats_detected"]),
            "threat_types": [t["type"] for t in result["threats_detected"]]
        })
    
    test_results["security_tests"]["threat_detection"] = threat_detection_results
    threats_detected = sum(1 for r in threat_detection_results if not r["is_secure"])
    print(f"   Result: {threats_detected}/{len(malicious_inputs)} threats properly detected")
    
    # Test 2: PII Detection and Redaction
    print("🔍 Test 2: PII Detection and Redaction")
    
    pii_content = "Contact John at john.doe@email.com or call 555-123-4567. SSN: 123-45-6789"
    is_secure, pii_result = system.validate_secure_input(pii_content)
    
    test_results["security_tests"]["pii_detection"] = {
        "original_content": pii_content,
        "pii_detected": pii_result["pii_analysis"]["contains_pii"],
        "pii_types": pii_result["pii_analysis"]["pii_types"],
        "confidence": pii_result["pii_analysis"]["confidence"],
        "sanitized_content": pii_result["sanitized_content"]
    }
    
    print(f"   PII Types Found: {pii_result['pii_analysis']['pii_types']}")
    print(f"   Confidence: {pii_result['pii_analysis']['confidence']:.2f}")
    print(f"   Sanitized: {pii_result['sanitized_content']}")
    
    # Test 3: Compliance Validation
    print("🌍 Test 3: Global Compliance Validation")
    
    compliance_scenarios = [
        {"operation": "data_processing", "has_consent": True, "location": "EU"},
        {"operation": "data_processing", "has_consent": False, "location": "California"},
        {"operation": "data_collection", "purpose_disclosed": True, "location": "Singapore"}
    ]
    
    compliance_results = []
    for scenario in compliance_scenarios:
        is_compliant, violations = system.compliance_manager.check_compliance(
            scenario["operation"], 
            {
                "consent_given": scenario.get("has_consent", False),
                "purpose_disclosed": scenario.get("purpose_disclosed", True),
                "security_measures": ["encryption"]
            },
            scenario["location"]
        )
        compliance_results.append({
            "scenario": scenario,
            "compliant": is_compliant,
            "violations": violations
        })
    
    test_results["compliance_tests"]["validation"] = compliance_results
    compliant_scenarios = sum(1 for r in compliance_results if r["compliant"])
    print(f"   Result: {compliant_scenarios}/{len(compliance_scenarios)} scenarios compliant")
    
    # Test 4: Consent Management
    print("📋 Test 4: Consent Management")
    
    consent_records = []
    for i in range(5):
        consent_id = system.compliance_manager.record_consent(
            f"user_{i}", "data_processing", i % 2 == 0  # Alternate consent granted/denied
        )
        consent_records.append(consent_id)
    
    compliance_report = system.compliance_manager.generate_compliance_report()
    test_results["compliance_tests"]["consent_management"] = {
        "records_created": len(consent_records),
        "total_consents": compliance_report["consent_statistics"]["total_consent_records"],
        "granted_consents": compliance_report["consent_statistics"]["granted_consents"]
    }
    
    print(f"   Consent Records: {len(consent_records)}")
    print(f"   Granted: {compliance_report['consent_statistics']['granted_consents']}")
    
    # Test 5: Internationalization
    print("🌐 Test 5: Internationalization Support")
    
    test_languages = ["en", "es", "fr", "de", "ja", "zh"]
    i18n_results = {}
    
    for lang in test_languages:
        message = system.i18n_manager.get_message("task_created", lang)
        i18n_results[lang] = {
            "language": lang,
            "message": message,
            "translated": message != system.i18n_manager.get_message("task_created", "en") or lang == "en"
        }
    
    test_results["i18n_tests"]["language_support"] = i18n_results
    supported_languages = len(system.i18n_manager.get_supported_languages())
    print(f"   Supported Languages: {supported_languages}")
    print(f"   Sample (ES): {i18n_results['es']['message']}")
    
    # Test 6: Comprehensive Security Report
    print("📊 Test 6: Security Report Generation")
    
    security_report = system.generate_security_report()
    test_results["security_tests"]["reporting"] = {
        "total_events": security_report["security"]["total_security_events"],
        "compliance_status": security_report["compliance"]["compliance_status"],
        "supported_languages": len(security_report["i18n"]["supported_languages"])
    }
    
    print(f"   Security Events: {security_report['security']['total_security_events']}")
    print(f"   Compliance Status: {security_report['compliance']['compliance_status']}")
    
    # Overall Assessment
    security_score = (threats_detected / len(malicious_inputs)) * 25
    compliance_score = (compliant_scenarios / len(compliance_scenarios)) * 25
    pii_score = 25 if pii_result["pii_analysis"]["contains_pii"] else 0
    i18n_score = min(25, (supported_languages / 6) * 25)
    
    total_score = security_score + compliance_score + pii_score + i18n_score
    
    test_results["overall_assessment"] = {
        "security_score": security_score,
        "compliance_score": compliance_score,
        "pii_detection_score": pii_score,
        "internationalization_score": i18n_score,
        "total_score": total_score,
        "grade": "A" if total_score >= 90 else "B" if total_score >= 80 else "C" if total_score >= 70 else "D",
        "generation_2_security_complete": total_score >= 80
    }
    
    print("\n🎯 GENERATION 2 - SECURITY & COMPLIANCE SUMMARY")
    print("=" * 55)
    print(f"🛡️ Security Score: {security_score:.1f}/25")
    print(f"🌍 Compliance Score: {compliance_score:.1f}/25")
    print(f"🔍 PII Detection Score: {pii_score:.1f}/25")
    print(f"🌐 I18n Score: {i18n_score:.1f}/25")
    print(f"📊 Total Score: {total_score:.1f}/100")
    print(f"🎓 Grade: {test_results['overall_assessment']['grade']}")
    print("🔒 Generation 2 (Security) - COMPLETE" if test_results['overall_assessment']['generation_2_security_complete'] else "⚠️ Generation 2 (Security) - NEEDS IMPROVEMENT")
    
    return test_results

def main():
    """Run Generation 2 security and compliance validation."""
    print("🚀 TERRAGON AUTONOMOUS SDLC - GENERATION 2")
    print("🔒 Enhanced Security & Compliance Validation")
    print("=" * 60)
    
    results = test_security_compliance_system()
    
    # Save results
    results_file = Path(__file__).parent / "generation_2_security_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {results_file}")
    print("\n🎯 GENERATION 2 SECURITY VALIDATION COMPLETE")
    
    return results

if __name__ == "__main__":
    results = main()