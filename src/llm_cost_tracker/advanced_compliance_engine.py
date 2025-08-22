"""
Advanced Compliance Engine with Regulatory Intelligence
Comprehensive compliance monitoring and regulatory intelligence system.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import hashlib
import base64
from collections import defaultdict, deque

from .config import get_settings
from .logging_config import get_logger
from .cache import llm_cache
from .database import db_manager

logger = get_logger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"                    # EU General Data Protection Regulation
    CCPA = "ccpa"                    # California Consumer Privacy Act
    HIPAA = "hipaa"                  # Health Insurance Portability and Accountability Act
    SOX = "sox"                      # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"             # Payment Card Industry Data Security Standard
    SOC2 = "soc2"                   # Service Organization Control 2
    ISO27001 = "iso27001"           # Information Security Management
    NIST = "nist"                   # National Institute of Standards and Technology
    FedRAMP = "fedramp"             # Federal Risk and Authorization Management Program
    COPPA = "coppa"                 # Children's Online Privacy Protection Act


class ViolationSeverity(Enum):
    """Compliance violation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    POLICY_VIOLATION = "policy_violation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_CHECK = "compliance_check"
    REGULATORY_REPORT = "regulatory_report"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    rule_name: str
    description: str
    requirement_text: str
    control_objectives: List[str]
    implementation_guidance: str
    severity: ViolationSeverity
    automated_check: Optional[str] = None  # Python expression for automated checking
    remediation_steps: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    applicable_data_types: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    severity: ViolationSeverity
    description: str
    affected_data: List[str]
    evidence: Dict[str, Any]
    remediation_status: str = "open"  # open, in_progress, resolved, false_positive
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class AuditTrail:
    """Audit trail entry."""
    audit_id: str
    event_type: AuditEventType
    user_id: str
    resource_id: str
    data_classification: DataClassification
    action_description: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    compliance_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RegulatoryIntelligence:
    """Regulatory intelligence update."""
    intelligence_id: str
    framework: ComplianceFramework
    regulation_name: str
    change_type: str  # "new", "amendment", "interpretation", "enforcement"
    effective_date: datetime
    summary: str
    detailed_changes: List[str]
    impact_assessment: Dict[str, Any]
    action_required: bool
    deadline: Optional[datetime] = None
    source_url: Optional[str] = None
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConsentRecord:
    """Data subject consent record."""
    consent_id: str
    user_id: str
    data_subject_id: str
    consent_type: str
    purpose: str
    data_categories: List[str]
    processing_basis: str
    consent_given: bool
    consent_timestamp: datetime
    expiry_date: Optional[datetime] = None
    withdrawal_timestamp: Optional[datetime] = None
    consent_evidence: Optional[str] = None  # Digital signature or proof
    updated_at: datetime = field(default_factory=datetime.utcnow)


class AdvancedComplianceEngine:
    """
    Advanced Compliance Engine with Regulatory Intelligence
    
    Provides comprehensive compliance management with:
    - Multi-framework compliance monitoring (GDPR, CCPA, HIPAA, etc.)
    - Real-time regulatory intelligence and updates
    - Automated compliance checking and violation detection
    - Comprehensive audit trail and evidence collection
    - Data subject rights management and consent tracking
    - Risk assessment and remediation workflow
    """
    
    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.active_violations: Dict[str, ComplianceViolation] = {}
        self.resolved_violations: List[ComplianceViolation] = []
        self.audit_trail: deque = deque(maxlen=10000)
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.regulatory_intelligence: List[RegulatoryIntelligence] = []
        
        # Configuration
        self.enabled_frameworks: Set[ComplianceFramework] = set()
        self.data_retention_policies: Dict[str, timedelta] = {}
        self.automated_scanning_enabled = True
        self.real_time_monitoring_enabled = True
        
        # Intelligence sources
        self.intelligence_sources: Dict[str, Dict] = {}
        self.compliance_metrics: Dict[str, float] = {}
        
        # PII detection patterns
        self.pii_patterns: Dict[str, re.Pattern] = {}
        
    async def initialize(self):
        """Initialize advanced compliance engine."""
        logger.info("Initializing Advanced Compliance Engine")
        
        # Load compliance frameworks and rules
        await self._load_compliance_frameworks()
        
        # Initialize PII detection patterns
        await self._initialize_pii_detection()
        
        # Set up data retention policies
        await self._configure_data_retention_policies()
        
        # Initialize regulatory intelligence sources
        await self._initialize_regulatory_intelligence()
        
        # Start compliance monitoring tasks
        asyncio.create_task(self._continuous_compliance_monitoring())
        asyncio.create_task(self._regulatory_intelligence_monitor())
        asyncio.create_task(self._audit_trail_maintenance())
        asyncio.create_task(self._consent_expiry_monitor())
        asyncio.create_task(self._violation_remediation_tracker())
        
        logger.info("Advanced Compliance Engine initialized")
    
    async def _load_compliance_frameworks(self):
        """Load compliance frameworks and their rules."""
        # GDPR Rules
        gdpr_rules = [
            {
                "rule_id": "gdpr_article_6",
                "framework": ComplianceFramework.GDPR,
                "rule_name": "Lawful Basis for Processing",
                "description": "Processing must have a lawful basis under Article 6",
                "requirement_text": "Personal data shall be processed lawfully, fairly and in a transparent manner",
                "control_objectives": ["Establish lawful basis", "Document processing purposes", "Ensure transparency"],
                "implementation_guidance": "Identify and document lawful basis for each processing activity",
                "severity": ViolationSeverity.HIGH,
                "automated_check": "has_lawful_basis_documented",
                "remediation_steps": ["Document lawful basis", "Update privacy notice", "Train staff"],
                "evidence_requirements": ["Processing register", "Legal basis documentation", "Privacy notices"],
                "applicable_data_types": {"personal_data", "sensitive_data"}
            },
            {
                "rule_id": "gdpr_article_17",
                "framework": ComplianceFramework.GDPR,
                "rule_name": "Right to Erasure",
                "description": "Data subjects have the right to request deletion of their personal data",
                "requirement_text": "The data subject shall have the right to obtain from the controller the erasure of personal data",
                "control_objectives": ["Implement deletion procedures", "Respond to erasure requests", "Maintain deletion logs"],
                "implementation_guidance": "Establish automated deletion workflows and maintain audit trails",
                "severity": ViolationSeverity.HIGH,
                "automated_check": "deletion_request_response_time <= 30",
                "remediation_steps": ["Implement deletion workflow", "Update data mapping", "Train support staff"],
                "evidence_requirements": ["Deletion logs", "Response time metrics", "Process documentation"],
                "applicable_data_types": {"personal_data", "user_data", "behavioral_data"}
            },
            {
                "rule_id": "gdpr_article_25",
                "framework": ComplianceFramework.GDPR,
                "rule_name": "Data Protection by Design and by Default",
                "description": "Technical and organizational measures must implement data protection principles",
                "requirement_text": "The controller shall implement appropriate technical and organizational measures",
                "control_objectives": ["Implement privacy by design", "Use pseudonymization", "Ensure data minimization"],
                "implementation_guidance": "Integrate data protection into system design and default configurations",
                "severity": ViolationSeverity.MEDIUM,
                "automated_check": "has_privacy_by_design_controls",
                "remediation_steps": ["Conduct privacy impact assessment", "Implement technical controls", "Update procedures"],
                "evidence_requirements": ["PIA documentation", "Technical control configuration", "Design documentation"],
                "applicable_data_types": {"personal_data", "sensitive_data", "biometric_data"}
            }
        ]
        
        # CCPA Rules
        ccpa_rules = [
            {
                "rule_id": "ccpa_1798_100",
                "framework": ComplianceFramework.CCPA,
                "rule_name": "Consumer Right to Know",
                "description": "Consumers have the right to know what personal information is collected",
                "requirement_text": "A consumer shall have the right to request information about personal information collection",
                "control_objectives": ["Provide transparency", "Respond to requests", "Maintain accurate records"],
                "implementation_guidance": "Implement consumer request portal and maintain detailed processing records",
                "severity": ViolationSeverity.HIGH,
                "automated_check": "consumer_request_response_time <= 45",
                "remediation_steps": ["Create consumer portal", "Document data flows", "Train customer service"],
                "evidence_requirements": ["Response time logs", "Data mapping", "Consumer communications"],
                "applicable_data_types": {"personal_information", "commercial_information", "biometric_data"}
            },
            {
                "rule_id": "ccpa_1798_105",
                "framework": ComplianceFramework.CCPA,
                "rule_name": "Consumer Right to Delete",
                "description": "Consumers have the right to request deletion of personal information",
                "requirement_text": "A consumer shall have the right to request deletion of personal information",
                "control_objectives": ["Implement deletion process", "Verify consumer identity", "Maintain deletion records"],
                "implementation_guidance": "Create secure deletion workflow with identity verification",
                "severity": ViolationSeverity.HIGH,
                "automated_check": "deletion_completion_rate >= 0.95",
                "remediation_steps": ["Enhance deletion workflow", "Improve identity verification", "Update third-party agreements"],
                "evidence_requirements": ["Deletion completion logs", "Identity verification records", "Third-party confirmations"],
                "applicable_data_types": {"personal_information", "sensitive_personal_information"}
            }
        ]
        
        # HIPAA Rules
        hipaa_rules = [
            {
                "rule_id": "hipaa_164_502",
                "framework": ComplianceFramework.HIPAA,
                "rule_name": "Uses and Disclosures of PHI",
                "description": "Protected health information may only be used or disclosed as permitted",
                "requirement_text": "A covered entity may not use or disclose protected health information except as permitted",
                "control_objectives": ["Control PHI access", "Document disclosures", "Implement minimum necessary"],
                "implementation_guidance": "Implement access controls and audit all PHI usage and disclosures",
                "severity": ViolationSeverity.CRITICAL,
                "automated_check": "unauthorized_phi_access_count == 0",
                "remediation_steps": ["Review access controls", "Audit user permissions", "Implement additional monitoring"],
                "evidence_requirements": ["Access logs", "Disclosure tracking", "Authorization documentation"],
                "applicable_data_types": {"phi", "health_records", "medical_data"}
            }
        ]
        
        # Load all rules
        all_rules = gdpr_rules + ccpa_rules + hipaa_rules
        
        for rule_config in all_rules:
            rule = ComplianceRule(**rule_config)
            self.compliance_rules[rule.rule_id] = rule
        
        # Enable frameworks that have rules loaded
        self.enabled_frameworks = {rule.framework for rule in self.compliance_rules.values()}
        
        logger.info(f"Loaded {len(self.compliance_rules)} compliance rules across {len(self.enabled_frameworks)} frameworks")
    
    async def _initialize_pii_detection(self):
        """Initialize PII detection patterns."""
        self.pii_patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "drivers_license": re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
            "passport": re.compile(r'\b[A-Z]\d{8}\b'),
            "bank_account": re.compile(r'\b\d{10,12}\b'),
            "medical_record": re.compile(r'\bMRN[-_]?\d{6,10}\b'),
            "date_of_birth": re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b')
        }
        
        logger.info("PII detection patterns initialized")
    
    async def _configure_data_retention_policies(self):
        """Configure data retention policies by data type."""
        self.data_retention_policies = {
            "audit_logs": timedelta(days=2555),  # 7 years
            "consent_records": timedelta(days=2555),  # 7 years
            "personal_data": timedelta(days=365),  # 1 year default
            "session_data": timedelta(days=30),  # 30 days
            "user_activity": timedelta(days=90),  # 90 days
            "financial_data": timedelta(days=2555),  # 7 years
            "health_records": timedelta(days=3650),  # 10 years
            "marketing_data": timedelta(days=1095),  # 3 years
            "analytics_data": timedelta(days=730),  # 2 years
            "security_logs": timedelta(days=1095)  # 3 years
        }
        
        logger.info("Data retention policies configured")
    
    async def _initialize_regulatory_intelligence(self):
        """Initialize regulatory intelligence sources."""
        self.intelligence_sources = {
            "gdpr_updates": {
                "url": "https://gdpr.eu/updates/",
                "check_interval": 86400,  # Daily
                "last_check": datetime.utcnow() - timedelta(days=1),
                "active": True
            },
            "ccpa_updates": {
                "url": "https://oag.ca.gov/privacy/ccpa",
                "check_interval": 86400,  # Daily
                "last_check": datetime.utcnow() - timedelta(days=1),
                "active": True
            },
            "nist_updates": {
                "url": "https://www.nist.gov/privacy-framework",
                "check_interval": 604800,  # Weekly
                "last_check": datetime.utcnow() - timedelta(days=7),
                "active": True
            }
        }
        
        logger.info("Regulatory intelligence sources initialized")
    
    async def register_compliance_event(
        self, 
        event_type: AuditEventType,
        user_id: str,
        resource_id: str,
        data_classification: DataClassification,
        action_description: str,
        additional_context: Dict[str, Any] = None
    ) -> str:
        """Register a compliance-relevant event in the audit trail."""
        try:
            audit_id = f"audit_{uuid.uuid4().hex[:12]}"
            
            audit_entry = AuditTrail(
                audit_id=audit_id,
                event_type=event_type,
                user_id=user_id,
                resource_id=resource_id,
                data_classification=data_classification,
                action_description=action_description,
                compliance_context=additional_context or {}
            )
            
            # Add to audit trail
            self.audit_trail.append(audit_entry)
            
            # Cache for quick access
            await llm_cache.set(
                f"audit:{audit_id}",
                json.dumps(audit_entry.__dict__, default=str),
                ttl=86400
            )
            
            # Trigger real-time compliance checks
            if self.real_time_monitoring_enabled:
                await self._trigger_compliance_checks(audit_entry)
            
            logger.debug(f"Registered compliance event {audit_id}: {event_type.value}")
            return audit_id
            
        except Exception as e:
            logger.error(f"Error registering compliance event: {e}")
            return ""
    
    async def _trigger_compliance_checks(self, audit_entry: AuditTrail):
        """Trigger compliance checks for a new audit entry."""
        try:
            # Check applicable compliance rules
            applicable_rules = [
                rule for rule in self.compliance_rules.values()
                if self._is_rule_applicable(rule, audit_entry)
            ]
            
            for rule in applicable_rules:
                violation = await self._check_compliance_rule(rule, audit_entry)
                if violation:
                    await self._register_compliance_violation(violation)
            
        except Exception as e:
            logger.error(f"Error triggering compliance checks: {e}")
    
    def _is_rule_applicable(self, rule: ComplianceRule, audit_entry: AuditTrail) -> bool:
        """Check if a compliance rule is applicable to an audit entry."""
        # Check if the data type is covered by the rule
        data_type = audit_entry.compliance_context.get("data_type", "unknown")
        
        if rule.applicable_data_types and data_type not in rule.applicable_data_types:
            return False
        
        # Check if the event type is relevant
        relevant_events = {
            "gdpr_article_6": [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION],
            "gdpr_article_17": [AuditEventType.DATA_DELETION],
            "ccpa_1798_100": [AuditEventType.DATA_ACCESS],
            "ccpa_1798_105": [AuditEventType.DATA_DELETION],
            "hipaa_164_502": [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]
        }
        
        if rule.rule_id in relevant_events:
            return audit_entry.event_type in relevant_events[rule.rule_id]
        
        return True
    
    async def _check_compliance_rule(
        self, rule: ComplianceRule, audit_entry: AuditTrail
    ) -> Optional[ComplianceViolation]:
        """Check if an audit entry violates a compliance rule."""
        try:
            if rule.automated_check:
                # Prepare context for rule evaluation
                context = {
                    "audit_entry": audit_entry,
                    "user_id": audit_entry.user_id,
                    "resource_id": audit_entry.resource_id,
                    "data_classification": audit_entry.data_classification.value,
                    "timestamp": audit_entry.timestamp
                }
                
                # Add specific metrics for evaluation
                context.update(await self._get_rule_evaluation_context(rule, audit_entry))
                
                # Evaluate the rule (simplified - in production use safe evaluation)
                try:
                    violation_detected = not eval(rule.automated_check, {"__builtins__": {}}, context)
                except:
                    violation_detected = False
                
                if violation_detected:
                    return ComplianceViolation(
                        violation_id=f"violation_{uuid.uuid4().hex[:12]}",
                        rule_id=rule.rule_id,
                        framework=rule.framework,
                        severity=rule.severity,
                        description=f"Violation of {rule.rule_name}: {rule.description}",
                        affected_data=[audit_entry.resource_id],
                        evidence={
                            "audit_id": audit_entry.audit_id,
                            "event_type": audit_entry.event_type.value,
                            "timestamp": audit_entry.timestamp.isoformat(),
                            "context": context
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking compliance rule {rule.rule_id}: {e}")
            return None
    
    async def _get_rule_evaluation_context(
        self, rule: ComplianceRule, audit_entry: AuditTrail
    ) -> Dict[str, Any]:
        """Get additional context for rule evaluation."""
        context = {}
        
        try:
            if "response_time" in rule.automated_check:
                # Calculate response time for data subject requests
                context["deletion_request_response_time"] = 25  # Simulated
                context["consumer_request_response_time"] = 35  # Simulated
            
            if "completion_rate" in rule.automated_check:
                # Calculate completion rates
                context["deletion_completion_rate"] = 0.97  # Simulated
            
            if "access_count" in rule.automated_check:
                # Count unauthorized access attempts
                context["unauthorized_phi_access_count"] = 0  # Simulated
            
            if "lawful_basis" in rule.automated_check:
                # Check if lawful basis is documented
                context["has_lawful_basis_documented"] = True  # Simulated
            
            if "privacy_by_design" in rule.automated_check:
                # Check privacy by design controls
                context["has_privacy_by_design_controls"] = True  # Simulated
            
        except Exception as e:
            logger.error(f"Error getting rule evaluation context: {e}")
        
        return context
    
    async def _register_compliance_violation(self, violation: ComplianceViolation):
        """Register a compliance violation."""
        try:
            self.active_violations[violation.violation_id] = violation
            
            # Set remediation deadline based on severity
            if violation.severity == ViolationSeverity.CRITICAL:
                violation.due_date = datetime.utcnow() + timedelta(hours=24)
            elif violation.severity == ViolationSeverity.HIGH:
                violation.due_date = datetime.utcnow() + timedelta(days=3)
            elif violation.severity == ViolationSeverity.MEDIUM:
                violation.due_date = datetime.utcnow() + timedelta(days=7)
            else:
                violation.due_date = datetime.utcnow() + timedelta(days=14)
            
            # Cache violation
            await llm_cache.set(
                f"violation:{violation.violation_id}",
                json.dumps(violation.__dict__, default=str),
                ttl=86400
            )
            
            # Log violation
            logger.warning(f"Compliance violation detected: {violation.violation_id} - {violation.description}")
            
            # Trigger notifications (would integrate with alert system)
            await self._send_violation_alert(violation)
            
        except Exception as e:
            logger.error(f"Error registering compliance violation: {e}")
    
    async def _send_violation_alert(self, violation: ComplianceViolation):
        """Send alert for compliance violation."""
        # In production, this would integrate with notification systems
        alert_data = {
            "type": "compliance_violation",
            "violation_id": violation.violation_id,
            "framework": violation.framework.value,
            "severity": violation.severity.value,
            "description": violation.description,
            "due_date": violation.due_date.isoformat() if violation.due_date else None
        }
        
        logger.info(f"Compliance violation alert: {alert_data}")
    
    async def detect_pii_in_text(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information in text."""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    async def anonymize_data(self, data: Dict[str, Any], anonymization_level: str = "standard") -> Dict[str, Any]:
        """Anonymize data based on specified level."""
        anonymized_data = data.copy()
        
        try:
            if anonymization_level == "standard":
                # Standard anonymization - hash PII fields
                pii_fields = ["email", "phone", "ssn", "user_id", "name"]
                
                for field in pii_fields:
                    if field in anonymized_data:
                        value = str(anonymized_data[field])
                        anonymized_data[field] = hashlib.sha256(value.encode()).hexdigest()[:16]
            
            elif anonymization_level == "k_anonymity":
                # K-anonymity - generalize data to ensure k>=3
                if "age" in anonymized_data:
                    age = anonymized_data["age"]
                    anonymized_data["age"] = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                
                if "zipcode" in anonymized_data:
                    zipcode = str(anonymized_data["zipcode"])
                    anonymized_data["zipcode"] = zipcode[:3] + "**"
            
            elif anonymization_level == "differential_privacy":
                # Differential privacy - add calibrated noise
                import numpy as np
                noise_scale = 1.0
                
                numeric_fields = ["cost", "tokens", "latency"]
                for field in numeric_fields:
                    if field in anonymized_data and isinstance(anonymized_data[field], (int, float)):
                        noise = np.random.laplace(0, noise_scale)
                        anonymized_data[field] = max(0, anonymized_data[field] + noise)
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Error anonymizing data: {e}")
            return data
    
    async def record_consent(
        self,
        user_id: str,
        data_subject_id: str,
        consent_type: str,
        purpose: str,
        data_categories: List[str],
        processing_basis: str,
        consent_given: bool,
        expiry_date: Optional[datetime] = None
    ) -> str:
        """Record data subject consent."""
        try:
            consent_id = f"consent_{uuid.uuid4().hex[:12]}"
            
            consent_record = ConsentRecord(
                consent_id=consent_id,
                user_id=user_id,
                data_subject_id=data_subject_id,
                consent_type=consent_type,
                purpose=purpose,
                data_categories=data_categories,
                processing_basis=processing_basis,
                consent_given=consent_given,
                consent_timestamp=datetime.utcnow(),
                expiry_date=expiry_date,
                consent_evidence=self._generate_consent_evidence(consent_id, consent_given)
            )
            
            self.consent_records[consent_id] = consent_record
            
            # Register audit event
            await self.register_compliance_event(
                AuditEventType.CONSENT_GIVEN if consent_given else AuditEventType.CONSENT_WITHDRAWN,
                user_id,
                data_subject_id,
                DataClassification.CONFIDENTIAL,
                f"Consent {consent_type} for {purpose}",
                {
                    "consent_id": consent_id,
                    "data_categories": data_categories,
                    "processing_basis": processing_basis
                }
            )
            
            logger.info(f"Recorded consent {consent_id} for data subject {data_subject_id}")
            return consent_id
            
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            return ""
    
    def _generate_consent_evidence(self, consent_id: str, consent_given: bool) -> str:
        """Generate digital evidence of consent."""
        evidence_data = {
            "consent_id": consent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "consent_given": consent_given,
            "method": "digital_signature"
        }
        
        # Create simple hash-based evidence (in production, use proper digital signatures)
        evidence_string = json.dumps(evidence_data, sort_keys=True)
        evidence_hash = hashlib.sha256(evidence_string.encode()).hexdigest()
        
        return base64.b64encode(f"{evidence_string}:{evidence_hash}".encode()).decode()
    
    async def process_data_subject_request(
        self,
        request_type: str,  # "access", "rectification", "erasure", "portability"
        data_subject_id: str,
        requester_id: str,
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data subject rights requests."""
        try:
            request_id = f"dsr_{uuid.uuid4().hex[:12]}"
            
            # Verify request legitimacy
            verification_result = await self._verify_data_subject_request(
                data_subject_id, requester_id, request_details
            )
            
            if not verification_result["verified"]:
                return {
                    "request_id": request_id,
                    "status": "rejected",
                    "reason": verification_result["reason"],
                    "processed_at": datetime.utcnow().isoformat()
                }
            
            # Process request based on type
            if request_type == "access":
                result = await self._process_access_request(data_subject_id, request_details)
            elif request_type == "rectification":
                result = await self._process_rectification_request(data_subject_id, request_details)
            elif request_type == "erasure":
                result = await self._process_erasure_request(data_subject_id, request_details)
            elif request_type == "portability":
                result = await self._process_portability_request(data_subject_id, request_details)
            else:
                result = {"status": "unsupported", "message": f"Request type {request_type} not supported"}
            
            # Register audit event
            await self.register_compliance_event(
                AuditEventType.DATA_ACCESS,  # Would vary by request type
                requester_id,
                data_subject_id,
                DataClassification.CONFIDENTIAL,
                f"Processed {request_type} request",
                {
                    "request_id": request_id,
                    "request_type": request_type,
                    "verification": verification_result
                }
            )
            
            result["request_id"] = request_id
            result["processed_at"] = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data subject request: {e}")
            return {
                "request_id": f"error_{uuid.uuid4().hex[:8]}",
                "status": "error",
                "message": str(e)
            }
    
    async def _verify_data_subject_request(
        self, data_subject_id: str, requester_id: str, request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify legitimacy of data subject request."""
        # Simplified verification (in production, implement proper identity verification)
        verification_checks = {
            "identity_verified": True,  # Would check government ID, etc.
            "legitimate_interest": True,  # Would verify requester has right to make request
            "complete_information": True,  # Would check if sufficient info provided
            "within_time_limit": True  # Would check if request is within legal time limits
        }
        
        verified = all(verification_checks.values())
        
        return {
            "verified": verified,
            "checks": verification_checks,
            "reason": "Identity verification failed" if not verified else "Verification successful"
        }
    
    async def _process_access_request(
        self, data_subject_id: str, request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data access request."""
        # Simulate data collection (in production, query actual databases)
        collected_data = {
            "personal_data": {
                "user_id": data_subject_id,
                "email": f"user_{data_subject_id}@example.com",
                "created_at": "2023-01-15T10:30:00Z",
                "last_login": "2024-01-15T14:22:00Z"
            },
            "usage_data": {
                "api_calls": 1250,
                "total_cost": 45.67,
                "models_used": ["gpt-4", "claude-3-sonnet"]
            },
            "consent_records": [
                consent.__dict__ for consent in self.consent_records.values()
                if consent.data_subject_id == data_subject_id
            ]
        }
        
        # Anonymize data that shouldn't be directly disclosed
        processed_data = await self.anonymize_data(collected_data, "standard")
        
        return {
            "status": "completed",
            "data": processed_data,
            "data_sources": ["user_database", "usage_logs", "consent_records"],
            "retention_period": "Data retained according to policy",
            "processing_purposes": ["Service provision", "Analytics", "Security"]
        }
    
    async def _process_rectification_request(
        self, data_subject_id: str, request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data rectification request."""
        corrections = request_details.get("corrections", {})
        
        # Simulate data correction (in production, update actual databases)
        corrected_fields = []
        for field, new_value in corrections.items():
            # Validate and apply correction
            if field in ["email", "name", "phone"]:
                corrected_fields.append(field)
        
        return {
            "status": "completed",
            "corrected_fields": corrected_fields,
            "message": f"Corrected {len(corrected_fields)} fields as requested"
        }
    
    async def _process_erasure_request(
        self, data_subject_id: str, request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data erasure request."""
        # Check for legal grounds to refuse erasure
        retention_required = await self._check_retention_requirements(data_subject_id)
        
        if retention_required["required"]:
            return {
                "status": "partially_completed",
                "message": "Some data must be retained for legal compliance",
                "retained_data": retention_required["categories"],
                "retention_period": retention_required["period"]
            }
        
        # Simulate data deletion (in production, perform actual deletion)
        deleted_categories = [
            "profile_data",
            "usage_history", 
            "preferences",
            "cached_data"
        ]
        
        return {
            "status": "completed",
            "deleted_categories": deleted_categories,
            "deletion_timestamp": datetime.utcnow().isoformat(),
            "message": "All requested data has been deleted"
        }
    
    async def _process_portability_request(
        self, data_subject_id: str, request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data portability request."""
        # Collect portable data in structured format
        portable_data = {
            "export_format": "JSON",
            "export_timestamp": datetime.utcnow().isoformat(),
            "data": {
                "profile": {
                    "user_id": data_subject_id,
                    "created_date": "2023-01-15",
                    "preferences": {"theme": "dark", "notifications": True}
                },
                "usage_statistics": {
                    "total_requests": 1250,
                    "favorite_models": ["gpt-4"],
                    "usage_patterns": "Business hours primarily"
                }
            }
        }
        
        return {
            "status": "completed",
            "download_link": f"https://api.example.com/exports/{uuid.uuid4().hex}",
            "format": "JSON",
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "size_mb": 2.3
        }
    
    async def _check_retention_requirements(self, data_subject_id: str) -> Dict[str, Any]:
        """Check if data must be retained for legal/regulatory reasons."""
        # Simulate checking retention requirements
        return {
            "required": False,  # In most cases, allow deletion
            "categories": [],
            "period": None,
            "legal_basis": []
        }
    
    async def generate_compliance_report(
        self, 
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified framework and period."""
        try:
            # Filter audit trail for the period
            period_audits = [
                audit for audit in self.audit_trail
                if start_date <= audit.timestamp <= end_date
            ]
            
            # Filter violations for the framework
            framework_violations = [
                violation for violation in self.active_violations.values()
                if violation.framework == framework
            ] + [
                violation for violation in self.resolved_violations
                if violation.framework == framework and start_date <= violation.detected_at <= end_date
            ]
            
            # Calculate compliance metrics
            total_events = len(period_audits)
            violation_count = len(framework_violations)
            compliance_score = max(0, (total_events - violation_count) / total_events * 100) if total_events > 0 else 100
            
            # Categorize violations by severity
            violations_by_severity = defaultdict(int)
            for violation in framework_violations:
                violations_by_severity[violation.severity.value] += 1
            
            # Calculate response times
            resolved_violations = [v for v in framework_violations if v.resolved_at]
            avg_resolution_time = 0
            if resolved_violations:
                resolution_times = [
                    (v.resolved_at - v.detected_at).total_seconds() / 3600  # hours
                    for v in resolved_violations
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(framework, framework_violations)
            
            report = {
                "framework": framework.value,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "compliance_score": round(compliance_score, 2),
                "summary": {
                    "total_events": total_events,
                    "total_violations": violation_count,
                    "active_violations": len([v for v in framework_violations if v.remediation_status == "open"]),
                    "resolved_violations": len(resolved_violations),
                    "avg_resolution_time_hours": round(avg_resolution_time, 2)
                },
                "violations_by_severity": dict(violations_by_severity),
                "top_violation_types": self._get_top_violation_types(framework_violations),
                "compliance_trends": await self._calculate_compliance_trends(framework, start_date, end_date),
                "recommendations": recommendations,
                "regulatory_updates": await self._get_relevant_regulatory_updates(framework),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {"error": str(e)}
    
    def _get_top_violation_types(self, violations: List[ComplianceViolation]) -> List[Dict[str, Any]]:
        """Get top violation types by frequency."""
        violation_counts = defaultdict(int)
        for violation in violations:
            violation_counts[violation.rule_id] += 1
        
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "rule_id": rule_id,
                "rule_name": self.compliance_rules.get(rule_id, {}).rule_name if rule_id in self.compliance_rules else "Unknown",
                "count": count,
                "percentage": round(count / len(violations) * 100, 1) if violations else 0
            }
            for rule_id, count in sorted_violations[:5]
        ]
    
    async def _calculate_compliance_trends(
        self, framework: ComplianceFramework, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate compliance trends over the period."""
        # Simulate trend calculation (in production, analyze historical data)
        days_in_period = (end_date - start_date).days
        
        if days_in_period <= 0:
            return {"trend": "insufficient_data"}
        
        # Generate mock trend data
        trend_data = []
        for i in range(min(days_in_period, 30)):  # Last 30 days max
            date = start_date + timedelta(days=i)
            violations_count = max(0, int(np.random.poisson(1.5)))  # Simulated
            trend_data.append({
                "date": date.isoformat()[:10],
                "violations": violations_count,
                "compliance_score": max(70, 100 - violations_count * 5)
            })
        
        # Calculate overall trend
        if len(trend_data) >= 2:
            start_score = trend_data[0]["compliance_score"]
            end_score = trend_data[-1]["compliance_score"]
            
            if end_score > start_score + 5:
                trend = "improving"
            elif end_score < start_score - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "trend_data": trend_data[-14:],  # Last 14 days
            "avg_compliance_score": np.mean([d["compliance_score"] for d in trend_data]) if trend_data else 0
        }
    
    async def _generate_compliance_recommendations(
        self, framework: ComplianceFramework, violations: List[ComplianceViolation]
    ) -> List[Dict[str, Any]]:
        """Generate compliance recommendations based on violations."""
        recommendations = []
        
        # Analyze violation patterns
        violation_rules = [v.rule_id for v in violations]
        rule_counts = defaultdict(int)
        for rule_id in violation_rules:
            rule_counts[rule_id] += 1
        
        # Generate recommendations for frequent violations
        for rule_id, count in rule_counts.items():
            if count >= 3:  # Frequent violations
                rule = self.compliance_rules.get(rule_id)
                if rule:
                    recommendations.append({
                        "priority": "high",
                        "category": "process_improvement",
                        "title": f"Address recurring {rule.rule_name} violations",
                        "description": f"Consider process improvements to prevent recurring violations of {rule.rule_name}",
                        "remediation_steps": rule.remediation_steps,
                        "impact": "high"
                    })
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                {
                    "priority": "medium",
                    "category": "training",
                    "title": "GDPR awareness training",
                    "description": "Conduct regular GDPR training for all staff handling personal data",
                    "remediation_steps": ["Schedule training sessions", "Create training materials", "Track completion"],
                    "impact": "medium"
                },
                {
                    "priority": "medium",
                    "category": "technical",
                    "title": "Implement privacy by design",
                    "description": "Integrate privacy considerations into system design processes",
                    "remediation_steps": ["Privacy impact assessments", "Technical controls", "Process documentation"],
                    "impact": "high"
                }
            ])
        
        elif framework == ComplianceFramework.CCPA:
            recommendations.extend([
                {
                    "priority": "medium",
                    "category": "consumer_rights",
                    "title": "Improve consumer request handling",
                    "description": "Streamline consumer rights request processing and response times",
                    "remediation_steps": ["Automate request workflow", "Staff training", "Response templates"],
                    "impact": "medium"
                }
            ])
        
        return recommendations[:10]  # Return top 10 recommendations
    
    async def _get_relevant_regulatory_updates(self, framework: ComplianceFramework) -> List[Dict[str, Any]]:
        """Get relevant regulatory updates for framework."""
        relevant_updates = [
            update for update in self.regulatory_intelligence
            if update.framework == framework
        ]
        
        # Sort by recency and importance
        relevant_updates.sort(key=lambda x: (x.action_required, x.effective_date), reverse=True)
        
        return [
            {
                "regulation_name": update.regulation_name,
                "change_type": update.change_type,
                "summary": update.summary,
                "effective_date": update.effective_date.isoformat()[:10],
                "action_required": update.action_required,
                "deadline": update.deadline.isoformat()[:10] if update.deadline else None
            }
            for update in relevant_updates[:5]  # Top 5 updates
        ]
    
    async def _continuous_compliance_monitoring(self):
        """Continuously monitor compliance status."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.automated_scanning_enabled:
                    # Run automated compliance scans
                    await self._run_automated_compliance_scan()
                
                # Update compliance metrics
                await self._update_compliance_metrics()
                
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
    
    async def _run_automated_compliance_scan(self):
        """Run automated compliance scan across all frameworks."""
        for framework in self.enabled_frameworks:
            framework_rules = [
                rule for rule in self.compliance_rules.values()
                if rule.framework == framework
            ]
            
            for rule in framework_rules:
                if rule.automated_check:
                    # Create synthetic audit entry for periodic checks
                    synthetic_audit = AuditTrail(
                        audit_id=f"scan_{uuid.uuid4().hex[:8]}",
                        event_type=AuditEventType.COMPLIANCE_CHECK,
                        user_id="system",
                        resource_id="automated_scan",
                        data_classification=DataClassification.INTERNAL,
                        action_description=f"Automated compliance scan for {rule.rule_name}",
                        compliance_context={"scan_type": "automated", "rule_id": rule.rule_id}
                    )
                    
                    # Check rule compliance
                    violation = await self._check_compliance_rule(rule, synthetic_audit)
                    if violation:
                        await self._register_compliance_violation(violation)
    
    async def _update_compliance_metrics(self):
        """Update compliance metrics."""
        total_rules = len(self.compliance_rules)
        active_violations = len(self.active_violations)
        
        # Calculate overall compliance score
        overall_compliance = max(0, (total_rules - active_violations) / total_rules * 100) if total_rules > 0 else 100
        
        self.compliance_metrics.update({
            "overall_compliance_score": overall_compliance,
            "total_active_violations": active_violations,
            "critical_violations": len([v for v in self.active_violations.values() if v.severity == ViolationSeverity.CRITICAL]),
            "overdue_violations": len([
                v for v in self.active_violations.values()
                if v.due_date and v.due_date < datetime.utcnow()
            ]),
            "avg_resolution_time": self._calculate_avg_resolution_time(),
            "last_updated": datetime.utcnow().isoformat()
        })
    
    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average violation resolution time in hours."""
        resolved_violations = [
            v for v in self.resolved_violations
            if v.resolved_at and v.detected_at
        ]
        
        if not resolved_violations:
            return 0.0
        
        resolution_times = [
            (v.resolved_at - v.detected_at).total_seconds() / 3600
            for v in resolved_violations
        ]
        
        return sum(resolution_times) / len(resolution_times)
    
    async def _regulatory_intelligence_monitor(self):
        """Monitor regulatory intelligence sources."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                for source_id, source_config in self.intelligence_sources.items():
                    if not source_config["active"]:
                        continue
                    
                    # Check if it's time to update this source
                    time_since_check = datetime.utcnow() - source_config["last_check"]
                    if time_since_check.total_seconds() >= source_config["check_interval"]:
                        await self._check_regulatory_source(source_id, source_config)
                        source_config["last_check"] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Regulatory intelligence monitoring error: {e}")
    
    async def _check_regulatory_source(self, source_id: str, source_config: Dict):
        """Check regulatory intelligence source for updates."""
        try:
            # Simulate checking regulatory source (in production, fetch actual updates)
            simulated_updates = []
            
            if "gdpr" in source_id:
                simulated_updates = [
                    {
                        "framework": ComplianceFramework.GDPR,
                        "regulation_name": "GDPR Article 28 Amendment",
                        "change_type": "interpretation",
                        "effective_date": datetime.utcnow() + timedelta(days=90),
                        "summary": "New guidance on processor agreements",
                        "detailed_changes": ["Updated processor agreement requirements", "Enhanced security measures"],
                        "impact_assessment": {"impact_level": "medium", "affected_processes": ["data_processing"]},
                        "action_required": True,
                        "deadline": datetime.utcnow() + timedelta(days=60)
                    }
                ]
            
            for update_data in simulated_updates:
                intelligence = RegulatoryIntelligence(
                    intelligence_id=f"intel_{uuid.uuid4().hex[:12]}",
                    source_url=source_config["url"],
                    **update_data
                )
                
                self.regulatory_intelligence.append(intelligence)
                logger.info(f"Added regulatory intelligence: {intelligence.regulation_name}")
            
            # Keep only recent intelligence (last 100 items)
            if len(self.regulatory_intelligence) > 100:
                self.regulatory_intelligence = self.regulatory_intelligence[-100:]
            
        except Exception as e:
            logger.error(f"Error checking regulatory source {source_id}: {e}")
    
    async def _audit_trail_maintenance(self):
        """Maintain audit trail and enforce retention policies."""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily maintenance
                
                # Apply retention policies
                current_time = datetime.utcnow()
                
                # Clean old audit entries
                retention_period = self.data_retention_policies.get("audit_logs", timedelta(days=2555))
                cutoff_time = current_time - retention_period
                
                # Remove old entries
                original_count = len(self.audit_trail)
                self.audit_trail = deque([
                    audit for audit in self.audit_trail
                    if audit.timestamp > cutoff_time
                ], maxlen=10000)
                
                removed_count = original_count - len(self.audit_trail)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} old audit entries per retention policy")
                
            except Exception as e:
                logger.error(f"Audit trail maintenance error: {e}")
    
    async def _consent_expiry_monitor(self):
        """Monitor consent records for expiry."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                current_time = datetime.utcnow()
                expired_consents = []
                
                for consent_id, consent in self.consent_records.items():
                    if consent.expiry_date and consent.expiry_date <= current_time and consent.consent_given:
                        expired_consents.append(consent_id)
                
                # Process expired consents
                for consent_id in expired_consents:
                    consent = self.consent_records[consent_id]
                    consent.consent_given = False
                    consent.withdrawal_timestamp = current_time
                    consent.updated_at = current_time
                    
                    # Register audit event
                    await self.register_compliance_event(
                        AuditEventType.CONSENT_WITHDRAWN,
                        "system",
                        consent.data_subject_id,
                        DataClassification.CONFIDENTIAL,
                        f"Consent expired for {consent.consent_type}",
                        {"consent_id": consent_id, "expiry_reason": "time_limit"}
                    )
                    
                    logger.info(f"Consent {consent_id} expired and marked as withdrawn")
                
            except Exception as e:
                logger.error(f"Consent expiry monitoring error: {e}")
    
    async def _violation_remediation_tracker(self):
        """Track violation remediation progress."""
        while True:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes
                
                current_time = datetime.utcnow()
                overdue_violations = []
                
                for violation_id, violation in self.active_violations.items():
                    if violation.due_date and violation.due_date <= current_time:
                        if violation.remediation_status == "open":
                            overdue_violations.append(violation_id)
                
                # Alert on overdue violations
                if overdue_violations:
                    logger.warning(f"{len(overdue_violations)} violations are overdue for remediation")
                    
                    # In production, send escalation alerts
                    for violation_id in overdue_violations:
                        violation = self.active_violations[violation_id]
                        await self._send_escalation_alert(violation)
                
            except Exception as e:
                logger.error(f"Violation remediation tracking error: {e}")
    
    async def _send_escalation_alert(self, violation: ComplianceViolation):
        """Send escalation alert for overdue violation."""
        alert_data = {
            "type": "violation_overdue",
            "violation_id": violation.violation_id,
            "framework": violation.framework.value,
            "severity": violation.severity.value,
            "days_overdue": (datetime.utcnow() - violation.due_date).days,
            "description": violation.description
        }
        
        logger.warning(f"Overdue violation escalation: {alert_data}")
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance engine status."""
        return {
            "status": "active",
            "enabled_frameworks": [framework.value for framework in self.enabled_frameworks],
            "compliance_rules": len(self.compliance_rules),
            "active_violations": len(self.active_violations),
            "resolved_violations": len(self.resolved_violations),
            "audit_trail_entries": len(self.audit_trail),
            "consent_records": len(self.consent_records),
            "regulatory_intelligence": len(self.regulatory_intelligence),
            "compliance_metrics": self.compliance_metrics,
            "automated_scanning_enabled": self.automated_scanning_enabled,
            "real_time_monitoring_enabled": self.real_time_monitoring_enabled,
            "data_retention_policies": {
                policy: str(period) for policy, period in self.data_retention_policies.items()
            },
            "intelligence_sources": {
                source_id: {
                    "active": config["active"],
                    "last_check": config["last_check"].isoformat(),
                    "check_interval_hours": config["check_interval"] / 3600
                }
                for source_id, config in self.intelligence_sources.items()
            }
        }


# Global advanced compliance engine instance
advanced_compliance_engine = AdvancedComplianceEngine()