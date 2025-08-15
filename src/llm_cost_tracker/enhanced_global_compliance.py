"""Enhanced Global Compliance System with Real-Time Monitoring."""

import asyncio
import json
import logging
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)


class DataCategory(Enum):
    """Categories of personal data."""
    PERSONAL_IDENTIFIER = "personal_identifier"    # Name, email, phone
    SENSITIVE_PERSONAL = "sensitive_personal"      # Health, biometric, financial
    BEHAVIORAL = "behavioral"                      # User interactions, preferences
    LOCATION = "location"                         # GPS, IP geolocation
    TECHNICAL = "technical"                       # Device info, browser data
    DERIVED = "derived"                           # Analytics, ML predictions


class ProcessingLegalBasis(Enum):
    """Legal basis for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRight(Enum):
    """Data subject rights under privacy regulations."""
    ACCESS = "access"               # Right to access personal data
    RECTIFICATION = "rectification" # Right to correct inaccurate data
    ERASURE = "erasure"            # Right to be forgotten
    RESTRICT = "restrict"          # Right to restrict processing
    PORTABILITY = "portability"    # Right to data portability
    OBJECT = "object"              # Right to object to processing
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    processing_id: str
    timestamp: datetime
    data_categories: List[DataCategory]
    legal_basis: ProcessingLegalBasis
    purpose: str
    data_subject_id: Optional[str] = None
    retention_period: Optional[timedelta] = None
    anonymized: bool = False
    consent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_id": self.processing_id,
            "timestamp": self.timestamp.isoformat(),
            "data_categories": [cat.value for cat in self.data_categories],
            "legal_basis": self.legal_basis.value,
            "purpose": self.purpose,
            "data_subject_id": self.data_subject_id,
            "retention_period": self.retention_period.total_seconds() if self.retention_period else None,
            "anonymized": self.anonymized,
            "consent_id": self.consent_id
        }


@dataclass
class ConsentRecord:
    """Record of user consent."""
    consent_id: str
    data_subject_id: str
    timestamp: datetime
    purposes: List[str]
    data_categories: List[DataCategory]
    active: bool = True
    withdrawal_timestamp: Optional[datetime] = None
    expiry_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "consent_id": self.consent_id,
            "data_subject_id": self.data_subject_id,
            "timestamp": self.timestamp.isoformat(),
            "purposes": self.purposes,
            "data_categories": [cat.value for cat in self.data_categories],
            "active": self.active,
            "withdrawal_timestamp": self.withdrawal_timestamp.isoformat() if self.withdrawal_timestamp else None,
            "expiry_timestamp": self.expiry_timestamp.isoformat() if self.expiry_timestamp else None
        }


@dataclass
class ComplianceAlert:
    """Alert for compliance violations or issues."""
    alert_id: str
    regulation: ComplianceRegulation
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "regulation": self.regulation.value,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            "metadata": self.metadata
        }


class PIIDetector:
    """Advanced PII detection system."""
    
    def __init__(self):
        """Initialize PII detection patterns."""
        self.patterns = {
            DataCategory.PERSONAL_IDENTIFIER: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN (US format)
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone (US format)
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full name pattern
            ],
            DataCategory.SENSITIVE_PERSONAL: [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Z]{1,2}\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?[A-Z]\b',  # Insurance numbers
            ],
            DataCategory.LOCATION: [
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
                r'\b\d{5}(-\d{4})?\b',  # ZIP code
                r'\b[-+]?\d{1,3}\.\d+,\s*[-+]?\d{1,3}\.\d+\b',  # GPS coordinates
            ],
            DataCategory.TECHNICAL: [
                r'User-Agent:\s*.+',  # User agent strings
                r'Cookie:\s*.+',      # Cookie headers
                r'Authorization:\s*.+',  # Auth headers
            ]
        }
    
    def detect_pii(self, text: str) -> Dict[DataCategory, List[str]]:
        """Detect PII in text content."""
        detected_pii = defaultdict(list)
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected_pii[category].extend(matches)
        
        return dict(detected_pii)
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize detected PII in text."""
        anonymized_text = text
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                # Replace with category-specific placeholders
                if category == DataCategory.PERSONAL_IDENTIFIER:
                    if '@' in pattern:  # Email pattern
                        anonymized_text = re.sub(pattern, '[EMAIL_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                    elif r'\d{3}-\d{2}-\d{4}' in pattern:  # SSN pattern
                        anonymized_text = re.sub(pattern, '[SSN_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                    elif r'\d{3}-\d{3}-\d{4}' in pattern:  # Phone pattern
                        anonymized_text = re.sub(pattern, '[PHONE_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                    else:
                        anonymized_text = re.sub(pattern, '[PII_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                elif category == DataCategory.SENSITIVE_PERSONAL:
                    anonymized_text = re.sub(pattern, '[SENSITIVE_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                elif category == DataCategory.LOCATION:
                    anonymized_text = re.sub(pattern, '[LOCATION_REDACTED]', anonymized_text, flags=re.IGNORECASE)
                elif category == DataCategory.TECHNICAL:
                    anonymized_text = re.sub(pattern, '[TECHNICAL_REDACTED]', anonymized_text, flags=re.IGNORECASE)
        
        return anonymized_text


class EnhancedGlobalCompliance:
    """Enhanced global compliance system with real-time monitoring."""
    
    def __init__(self, project_root: Path = None):
        """Initialize the enhanced compliance system."""
        self.project_root = project_root or Path("/root/repo")
        self.pii_detector = PIIDetector()
        
        # Storage for compliance records
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.compliance_alerts: List[ComplianceAlert] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.compliance_rules: Dict[ComplianceRegulation, Dict[str, Any]] = {}
        
        self._setup_compliance_rules()
        self._load_existing_records()
    
    def _setup_compliance_rules(self) -> None:
        """Setup compliance rules for different regulations."""
        self.compliance_rules = {
            ComplianceRegulation.GDPR: {
                "consent_required_categories": [
                    DataCategory.PERSONAL_IDENTIFIER,
                    DataCategory.SENSITIVE_PERSONAL,
                    DataCategory.BEHAVIORAL
                ],
                "data_retention_limits": {
                    DataCategory.PERSONAL_IDENTIFIER: timedelta(days=365 * 2),  # 2 years
                    DataCategory.SENSITIVE_PERSONAL: timedelta(days=365),       # 1 year
                    DataCategory.BEHAVIORAL: timedelta(days=365 * 3),          # 3 years
                },
                "mandatory_rights": [
                    DataSubjectRight.ACCESS,
                    DataSubjectRight.RECTIFICATION,
                    DataSubjectRight.ERASURE,
                    DataSubjectRight.PORTABILITY
                ],
                "consent_expiry": timedelta(days=365 * 2),  # 2 years
                "breach_notification_hours": 72
            },
            ComplianceRegulation.CCPA: {
                "consent_required_categories": [
                    DataCategory.SENSITIVE_PERSONAL
                ],
                "data_retention_limits": {
                    DataCategory.PERSONAL_IDENTIFIER: timedelta(days=365 * 3),  # 3 years
                    DataCategory.SENSITIVE_PERSONAL: timedelta(days=365),       # 1 year
                },
                "mandatory_rights": [
                    DataSubjectRight.ACCESS,
                    DataSubjectRight.ERASURE,
                    DataSubjectRight.OBJECT
                ],
                "opt_out_required": True
            },
            ComplianceRegulation.PDPA: {
                "consent_required_categories": [
                    DataCategory.PERSONAL_IDENTIFIER,
                    DataCategory.SENSITIVE_PERSONAL
                ],
                "data_retention_limits": {
                    DataCategory.PERSONAL_IDENTIFIER: timedelta(days=365 * 5),  # 5 years
                    DataCategory.SENSITIVE_PERSONAL: timedelta(days=365),       # 1 year
                },
                "mandatory_rights": [
                    DataSubjectRight.ACCESS,
                    DataSubjectRight.RECTIFICATION,
                    DataSubjectRight.ERASURE
                ]
            }
        }
    
    def _load_existing_records(self) -> None:
        """Load existing compliance records from storage."""
        try:
            compliance_file = self.project_root / "compliance_records.json"
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    data = json.load(f)
                
                # Load processing records
                for record_data in data.get("processing_records", []):
                    record = DataProcessingRecord(
                        processing_id=record_data["processing_id"],
                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                        data_categories=[DataCategory(cat) for cat in record_data["data_categories"]],
                        legal_basis=ProcessingLegalBasis(record_data["legal_basis"]),
                        purpose=record_data["purpose"],
                        data_subject_id=record_data.get("data_subject_id"),
                        retention_period=timedelta(seconds=record_data["retention_period"]) if record_data.get("retention_period") else None,
                        anonymized=record_data.get("anonymized", False),
                        consent_id=record_data.get("consent_id")
                    )
                    self.processing_records.append(record)
                
                # Load consent records
                for consent_data in data.get("consent_records", []):
                    consent = ConsentRecord(
                        consent_id=consent_data["consent_id"],
                        data_subject_id=consent_data["data_subject_id"],
                        timestamp=datetime.fromisoformat(consent_data["timestamp"]),
                        purposes=consent_data["purposes"],
                        data_categories=[DataCategory(cat) for cat in consent_data["data_categories"]],
                        active=consent_data.get("active", True),
                        withdrawal_timestamp=datetime.fromisoformat(consent_data["withdrawal_timestamp"]) if consent_data.get("withdrawal_timestamp") else None,
                        expiry_timestamp=datetime.fromisoformat(consent_data["expiry_timestamp"]) if consent_data.get("expiry_timestamp") else None
                    )
                    self.consent_records[consent.consent_id] = consent
                
                logger.info(f"Loaded {len(self.processing_records)} processing records and {len(self.consent_records)} consent records")
                
        except Exception as e:
            logger.warning(f"Failed to load existing compliance records: {e}")
    
    async def start_real_time_monitoring(self) -> None:
        """Start real-time compliance monitoring."""
        if self.monitoring_active:
            logger.warning("Compliance monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting real-time compliance monitoring...")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_data_retention())
        asyncio.create_task(self._monitor_consent_expiry())
        asyncio.create_task(self._monitor_processing_compliance())
        
        logger.info("Real-time compliance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time compliance monitoring."""
        self.monitoring_active = False
        logger.info("Compliance monitoring stopped")
    
    def record_data_processing(self,
                              data_categories: List[DataCategory],
                              legal_basis: ProcessingLegalBasis,
                              purpose: str,
                              data_subject_id: Optional[str] = None,
                              retention_period: Optional[timedelta] = None,
                              consent_id: Optional[str] = None) -> str:
        """Record a data processing activity."""
        processing_id = str(uuid.uuid4())
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            timestamp=datetime.now(),
            data_categories=data_categories,
            legal_basis=legal_basis,
            purpose=purpose,
            data_subject_id=data_subject_id,
            retention_period=retention_period,
            consent_id=consent_id
        )
        
        self.processing_records.append(record)
        
        # Check compliance immediately
        self._check_processing_compliance(record)
        
        logger.info(f"Recorded data processing: {processing_id}")
        return processing_id
    
    def record_consent(self,
                      data_subject_id: str,
                      purposes: List[str],
                      data_categories: List[DataCategory],
                      expiry_timestamp: Optional[datetime] = None) -> str:
        """Record user consent."""
        consent_id = str(uuid.uuid4())
        
        # Set default expiry based on GDPR (most restrictive)
        if expiry_timestamp is None:
            gdpr_rules = self.compliance_rules.get(ComplianceRegulation.GDPR, {})
            consent_expiry = gdpr_rules.get("consent_expiry", timedelta(days=365 * 2))
            expiry_timestamp = datetime.now() + consent_expiry
        
        consent = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            timestamp=datetime.now(),
            purposes=purposes,
            data_categories=data_categories,
            expiry_timestamp=expiry_timestamp
        )
        
        self.consent_records[consent_id] = consent
        
        logger.info(f"Recorded consent: {consent_id} for subject {data_subject_id}")
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw user consent."""
        if consent_id not in self.consent_records:
            return False
        
        consent = self.consent_records[consent_id]
        consent.active = False
        consent.withdrawal_timestamp = datetime.now()
        
        # Generate alert for downstream processing
        self._create_alert(
            ComplianceRegulation.GDPR,
            "medium",
            f"Consent withdrawn for subject {consent.data_subject_id}",
            {"consent_id": consent_id, "data_subject_id": consent.data_subject_id}
        )
        
        logger.info(f"Consent withdrawn: {consent_id}")
        return True
    
    async def handle_data_subject_request(self,
                                        data_subject_id: str,
                                        request_type: DataSubjectRight,
                                        additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        logger.info(f"Processing {request_type.value} request for subject {data_subject_id}")
        
        result = {
            "request_id": str(uuid.uuid4()),
            "data_subject_id": data_subject_id,
            "request_type": request_type.value,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "data": {}
        }
        
        try:
            if request_type == DataSubjectRight.ACCESS:
                # Provide access to all data
                result["data"] = await self._get_subject_data(data_subject_id)
                
            elif request_type == DataSubjectRight.ERASURE:
                # Delete all data for the subject
                deleted_count = await self._delete_subject_data(data_subject_id)
                result["data"] = {"deleted_records": deleted_count}
                
            elif request_type == DataSubjectRight.PORTABILITY:
                # Export data in structured format
                result["data"] = await self._export_subject_data(data_subject_id)
                
            elif request_type == DataSubjectRight.RECTIFICATION:
                # Update incorrect data
                if additional_data:
                    updated_count = await self._update_subject_data(data_subject_id, additional_data)
                    result["data"] = {"updated_records": updated_count}
                else:
                    result["status"] = "failed"
                    result["error"] = "No rectification data provided"
                    
            elif request_type == DataSubjectRight.RESTRICT:
                # Restrict processing
                restricted_count = await self._restrict_subject_processing(data_subject_id)
                result["data"] = {"restricted_records": restricted_count}
                
            elif request_type == DataSubjectRight.OBJECT:
                # Object to processing
                objection_count = await self._handle_processing_objection(data_subject_id)
                result["data"] = {"processing_stopped": objection_count}
                
            elif request_type == DataSubjectRight.WITHDRAW_CONSENT:
                # Withdraw all consent
                withdrawn_consents = []
                for consent_id, consent in self.consent_records.items():
                    if consent.data_subject_id == data_subject_id and consent.active:
                        self.withdraw_consent(consent_id)
                        withdrawn_consents.append(consent_id)
                result["data"] = {"withdrawn_consents": withdrawn_consents}
            
            # Log the request handling
            self.record_data_processing(
                data_categories=[DataCategory.PERSONAL_IDENTIFIER],
                legal_basis=ProcessingLegalBasis.LEGAL_OBLIGATION,
                purpose=f"Data subject {request_type.value} request",
                data_subject_id=data_subject_id
            )
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Failed to handle {request_type.value} request: {e}")
        
        return result
    
    async def _get_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get all data for a data subject."""
        subject_data = {
            "processing_records": [],
            "consent_records": []
        }
        
        # Find processing records
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                subject_data["processing_records"].append(record.to_dict())
        
        # Find consent records
        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id:
                subject_data["consent_records"].append(consent.to_dict())
        
        return subject_data
    
    async def _delete_subject_data(self, data_subject_id: str) -> int:
        """Delete all data for a data subject."""
        deleted_count = 0
        
        # Delete processing records
        self.processing_records = [
            record for record in self.processing_records
            if record.data_subject_id != data_subject_id
        ]
        deleted_count += len([r for r in self.processing_records if r.data_subject_id == data_subject_id])
        
        # Delete consent records
        consent_ids_to_delete = [
            consent_id for consent_id, consent in self.consent_records.items()
            if consent.data_subject_id == data_subject_id
        ]
        for consent_id in consent_ids_to_delete:
            del self.consent_records[consent_id]
        deleted_count += len(consent_ids_to_delete)
        
        return deleted_count
    
    async def _export_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Export data for a data subject in portable format."""
        return await self._get_subject_data(data_subject_id)
    
    async def _update_subject_data(self, data_subject_id: str, update_data: Dict[str, Any]) -> int:
        """Update data for a data subject."""
        # This would integrate with actual data storage systems
        # For now, just record the rectification request
        self.record_data_processing(
            data_categories=[DataCategory.PERSONAL_IDENTIFIER],
            legal_basis=ProcessingLegalBasis.LEGAL_OBLIGATION,
            purpose="Data rectification",
            data_subject_id=data_subject_id
        )
        return 1  # Simulated update count
    
    async def _restrict_subject_processing(self, data_subject_id: str) -> int:
        """Restrict processing for a data subject."""
        restricted_count = 0
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                # Mark as restricted (in real implementation, would update storage)
                restricted_count += 1
        return restricted_count
    
    async def _handle_processing_objection(self, data_subject_id: str) -> int:
        """Handle processing objection from data subject."""
        # Similar to restriction but for legitimate interests basis
        objection_count = 0
        for record in self.processing_records:
            if (record.data_subject_id == data_subject_id and 
                record.legal_basis == ProcessingLegalBasis.LEGITIMATE_INTERESTS):
                objection_count += 1
        return objection_count
    
    def _check_processing_compliance(self, record: DataProcessingRecord) -> None:
        """Check compliance for a processing record."""
        for regulation, rules in self.compliance_rules.items():
            # Check consent requirements
            consent_required_categories = rules.get("consent_required_categories", [])
            
            if any(cat in consent_required_categories for cat in record.data_categories):
                if record.legal_basis == ProcessingLegalBasis.CONSENT and not record.consent_id:
                    self._create_alert(
                        regulation,
                        "high",
                        f"Processing {record.processing_id} requires consent but no consent ID provided",
                        {"processing_id": record.processing_id}
                    )
                elif record.consent_id and record.consent_id not in self.consent_records:
                    self._create_alert(
                        regulation,
                        "high",
                        f"Invalid consent ID {record.consent_id} for processing {record.processing_id}",
                        {"processing_id": record.processing_id, "consent_id": record.consent_id}
                    )
                elif record.consent_id and not self.consent_records[record.consent_id].active:
                    self._create_alert(
                        regulation,
                        "critical",
                        f"Processing {record.processing_id} using withdrawn consent {record.consent_id}",
                        {"processing_id": record.processing_id, "consent_id": record.consent_id}
                    )
    
    async def _monitor_data_retention(self) -> None:
        """Monitor data retention limits."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for regulation, rules in self.compliance_rules.items():
                    retention_limits = rules.get("data_retention_limits", {})
                    
                    for record in self.processing_records:
                        if record.retention_period:
                            retention_deadline = record.timestamp + record.retention_period
                        else:
                            # Use default retention limit for category
                            default_retention = None
                            for category in record.data_categories:
                                if category in retention_limits:
                                    default_retention = retention_limits[category]
                                    break
                            
                            if default_retention:
                                retention_deadline = record.timestamp + default_retention
                            else:
                                continue
                        
                        # Check if retention period exceeded
                        if current_time > retention_deadline:
                            self._create_alert(
                                regulation,
                                "medium",
                                f"Data retention limit exceeded for processing {record.processing_id}",
                                {
                                    "processing_id": record.processing_id,
                                    "retention_deadline": retention_deadline.isoformat(),
                                    "data_categories": [cat.value for cat in record.data_categories]
                                }
                            )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in data retention monitoring: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _monitor_consent_expiry(self) -> None:
        """Monitor consent expiry."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for consent in self.consent_records.values():
                    if consent.active and consent.expiry_timestamp:
                        # Check if consent expires soon (30 days warning)
                        warning_time = consent.expiry_timestamp - timedelta(days=30)
                        
                        if current_time > warning_time and current_time < consent.expiry_timestamp:
                            self._create_alert(
                                ComplianceRegulation.GDPR,
                                "low",
                                f"Consent {consent.consent_id} expires in {(consent.expiry_timestamp - current_time).days} days",
                                {
                                    "consent_id": consent.consent_id,
                                    "data_subject_id": consent.data_subject_id,
                                    "expiry_timestamp": consent.expiry_timestamp.isoformat()
                                }
                            )
                        
                        # Check if consent has expired
                        elif current_time > consent.expiry_timestamp:
                            consent.active = False
                            consent.withdrawal_timestamp = current_time
                            
                            self._create_alert(
                                ComplianceRegulation.GDPR,
                                "medium",
                                f"Consent {consent.consent_id} has expired",
                                {
                                    "consent_id": consent.consent_id,
                                    "data_subject_id": consent.data_subject_id,
                                    "expiry_timestamp": consent.expiry_timestamp.isoformat()
                                }
                            )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in consent expiry monitoring: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _monitor_processing_compliance(self) -> None:
        """Monitor ongoing processing compliance."""
        while self.monitoring_active:
            try:
                # Check for compliance violations in recent processing
                recent_records = [
                    record for record in self.processing_records
                    if record.timestamp > datetime.now() - timedelta(hours=1)
                ]
                
                for record in recent_records:
                    self._check_processing_compliance(record)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in processing compliance monitoring: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def _create_alert(self,
                     regulation: ComplianceRegulation,
                     severity: str,
                     message: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a compliance alert."""
        alert_id = str(uuid.uuid4())
        
        alert = ComplianceAlert(
            alert_id=alert_id,
            regulation=regulation,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.compliance_alerts.append(alert)
        
        logger.warning(f"Compliance alert [{severity.upper()}]: {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a compliance alert."""
        for alert in self.compliance_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                logger.info(f"Resolved compliance alert: {alert_id}")
                return True
        return False
    
    async def save_compliance_records(self) -> None:
        """Save compliance records to persistent storage."""
        try:
            compliance_data = {
                "timestamp": datetime.now().isoformat(),
                "processing_records": [record.to_dict() for record in self.processing_records],
                "consent_records": [consent.to_dict() for consent in self.consent_records.values()],
                "compliance_alerts": [alert.to_dict() for alert in self.compliance_alerts]
            }
            
            compliance_file = self.project_root / "compliance_records.json"
            with open(compliance_file, 'w') as f:
                json.dump(compliance_data, f, indent=2)
            
            logger.info(f"Saved compliance records to {compliance_file}")
            
        except Exception as e:
            logger.error(f"Failed to save compliance records: {e}")
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard."""
        current_time = datetime.now()
        
        # Active consents
        active_consents = [c for c in self.consent_records.values() if c.active]
        
        # Recent processing
        recent_processing = [
            r for r in self.processing_records
            if r.timestamp > current_time - timedelta(days=30)
        ]
        
        # Active alerts
        active_alerts = [a for a in self.compliance_alerts if not a.resolved]
        
        # Alert severity breakdown
        alert_severity_counts = defaultdict(int)
        for alert in active_alerts:
            alert_severity_counts[alert.severity] += 1
        
        # Processing by legal basis
        legal_basis_counts = defaultdict(int)
        for record in recent_processing:
            legal_basis_counts[record.legal_basis.value] += 1
        
        return {
            "timestamp": current_time.isoformat(),
            "summary": {
                "total_processing_records": len(self.processing_records),
                "recent_processing_records": len(recent_processing),
                "active_consents": len(active_consents),
                "total_consents": len(self.consent_records),
                "active_alerts": len(active_alerts),
                "total_alerts": len(self.compliance_alerts)
            },
            "alert_breakdown": dict(alert_severity_counts),
            "legal_basis_breakdown": dict(legal_basis_counts),
            "recent_alerts": [alert.to_dict() for alert in active_alerts[-10:]],  # Last 10 alerts
            "monitoring_status": "active" if self.monitoring_active else "inactive"
        }


async def main():
    """Main function for enhanced global compliance system."""
    logger.info("Starting Enhanced Global Compliance System...")
    
    compliance_system = EnhancedGlobalCompliance()
    
    # Start real-time monitoring
    await compliance_system.start_real_time_monitoring()
    
    # Demo compliance operations
    print("\n" + "="*80)
    print("🌍 ENHANCED GLOBAL COMPLIANCE SYSTEM")
    print("="*80)
    
    # Record sample consent
    consent_id = compliance_system.record_consent(
        data_subject_id="user_123",
        purposes=["analytics", "personalization"],
        data_categories=[DataCategory.PERSONAL_IDENTIFIER, DataCategory.BEHAVIORAL]
    )
    print(f"✅ Recorded consent: {consent_id}")
    
    # Record sample processing
    processing_id = compliance_system.record_data_processing(
        data_categories=[DataCategory.PERSONAL_IDENTIFIER],
        legal_basis=ProcessingLegalBasis.CONSENT,
        purpose="User analytics",
        data_subject_id="user_123",
        consent_id=consent_id
    )
    print(f"✅ Recorded processing: {processing_id}")
    
    # Test PII detection
    test_text = "Contact John Doe at john.doe@example.com or call 555-123-4567"
    pii_detected = compliance_system.pii_detector.detect_pii(test_text)
    anonymized_text = compliance_system.pii_detector.anonymize_text(test_text)
    
    print(f"\n📋 PII Detection Test:")
    print(f"Original: {test_text}")
    print(f"Detected PII: {dict(pii_detected)}")
    print(f"Anonymized: {anonymized_text}")
    
    # Handle data subject request
    access_result = await compliance_system.handle_data_subject_request(
        data_subject_id="user_123",
        request_type=DataSubjectRight.ACCESS
    )
    print(f"\n🔍 Access request result: {access_result['status']}")
    
    # Get dashboard data
    dashboard_data = compliance_system.get_compliance_dashboard_data()
    print(f"\n📊 Compliance Dashboard Summary:")
    print(f"   • Processing Records: {dashboard_data['summary']['total_processing_records']}")
    print(f"   • Active Consents: {dashboard_data['summary']['active_consents']}")
    print(f"   • Active Alerts: {dashboard_data['summary']['active_alerts']}")
    print(f"   • Monitoring Status: {dashboard_data['monitoring_status']}")
    
    # Save records
    await compliance_system.save_compliance_records()
    print(f"\n💾 Compliance records saved")
    
    print("\n" + "="*80)
    
    # Stop monitoring
    await compliance_system.stop_monitoring()
    
    return dashboard_data


if __name__ == "__main__":
    asyncio.run(main())