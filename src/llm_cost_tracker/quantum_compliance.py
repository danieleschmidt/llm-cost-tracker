"""
Compliance and data protection features for Quantum Task Planner.
Provides GDPR, CCPA, PDPA compliance and data governance capabilities.
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    SENSITIVE = "sensitive"  # Sensitive personal data


class ComplianceRegion(Enum):
    """Compliance regions and frameworks."""

    EU_GDPR = "eu_gdpr"  # European Union - General Data Protection Regulation
    US_CCPA = "us_ccpa"  # California Consumer Privacy Act
    SG_PDPA = "sg_pdpa"  # Singapore Personal Data Protection Act
    UK_GDPR = "uk_gdpr"  # UK GDPR
    CANADA_PIPEDA = (
        "ca_pipeda"  # Personal Information Protection and Electronic Documents Act
    )
    BRAZIL_LGPD = "br_lgpd"  # Lei Geral de Proteção de Dados
    GLOBAL = "global"  # Global best practices


class DataRetentionPolicy(Enum):
    """Data retention policies."""

    IMMEDIATE = "immediate"  # Delete immediately after processing
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"  # Keep indefinitely
    CUSTOM = "custom"  # Custom retention period


@dataclass
class PIIField:
    """Configuration for a PII field."""

    field_name: str
    classification: DataClassification
    retention_policy: DataRetentionPolicy
    anonymization_method: str = "hash"  # hash, mask, encrypt, delete
    is_required: bool = False
    custom_retention_days: Optional[int] = None

    def get_retention_days(self) -> int:
        """Get retention period in days."""
        if self.retention_policy == DataRetentionPolicy.IMMEDIATE:
            return 0
        elif self.retention_policy == DataRetentionPolicy.SHORT_TERM:
            return 30
        elif self.retention_policy == DataRetentionPolicy.MEDIUM_TERM:
            return 365
        elif self.retention_policy == DataRetentionPolicy.LONG_TERM:
            return 365 * 7
        elif self.retention_policy == DataRetentionPolicy.PERMANENT:
            return -1  # Never delete
        elif self.retention_policy == DataRetentionPolicy.CUSTOM:
            return self.custom_retention_days or 30
        else:
            return 30  # Default


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""

    record_id: str
    timestamp: datetime
    data_subject_id: Optional[str]
    processing_purpose: str
    data_categories: List[DataClassification]
    legal_basis: str
    retention_period: int
    data_controller: str
    data_processor: str
    third_party_transfers: List[str] = field(default_factory=list)
    automated_decision_making: bool = False
    profiling: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "data_subject_id": self.data_subject_id,
            "processing_purpose": self.processing_purpose,
            "data_categories": [cat.value for cat in self.data_categories],
            "legal_basis": self.legal_basis,
            "retention_period": self.retention_period,
            "data_controller": self.data_controller,
            "data_processor": self.data_processor,
            "third_party_transfers": self.third_party_transfers,
            "automated_decision_making": self.automated_decision_making,
            "profiling": self.profiling,
        }


@dataclass
class ConsentRecord:
    """Record of user consent."""

    consent_id: str
    data_subject_id: str
    timestamp: datetime
    consent_given: bool
    purposes: List[str]
    lawful_basis: str
    consent_method: str  # explicit, implicit, opt_in, opt_out
    withdrawal_timestamp: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if not self.consent_given:
            return False
        if self.withdrawal_timestamp:
            return False

        # Consent should be refreshed periodically (e.g., every 2 years)
        if (datetime.now() - self.timestamp).days > 730:
            return False

        return True


class PIIDetector:
    """Detects and classifies PII in text data."""

    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(
                r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
            ),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            "name": re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),  # Simple name pattern
        }

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return matches by type."""
        if not isinstance(text, str):
            return {}

        detected = {}
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches

        return detected

    def has_pii(self, text: str) -> bool:
        """Check if text contains any PII."""
        return bool(self.detect_pii(text))

    def classify_text(self, text: str) -> DataClassification:
        """Classify text based on PII content."""
        pii_detected = self.detect_pii(text)

        if any(pii_type in ["ssn", "credit_card"] for pii_type in pii_detected):
            return DataClassification.RESTRICTED
        elif any(pii_type in ["email", "phone", "name"] for pii_type in pii_detected):
            return DataClassification.PII
        elif pii_detected:
            return DataClassification.SENSITIVE
        else:
            return DataClassification.INTERNAL


class DataAnonymizer:
    """Anonymizes and pseudonymizes sensitive data."""

    def __init__(self, salt: str = "quantum_task_planner_salt"):
        self.salt = salt
        self.pii_detector = PIIDetector()

    def hash_data(self, data: str) -> str:
        """Hash data with salt for anonymization."""
        combined = f"{data}{self.salt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def mask_data(self, data: str, mask_char: str = "*", keep_chars: int = 2) -> str:
        """Mask data showing only first/last characters."""
        if len(data) <= keep_chars * 2:
            return mask_char * len(data)

        return (
            data[:keep_chars]
            + mask_char * (len(data) - keep_chars * 2)
            + data[-keep_chars:]
        )

    def anonymize_text(self, text: str, method: str = "hash") -> str:
        """Anonymize PII in text."""
        if not isinstance(text, str):
            return str(text)

        pii_detected = self.pii_detector.detect_pii(text)
        anonymized_text = text

        for pii_type, matches in pii_detected.items():
            for match in matches:
                if method == "hash":
                    replacement = f"[{pii_type.upper()}_{self.hash_data(match)[:8]}]"
                elif method == "mask":
                    replacement = self.mask_data(match)
                elif method == "delete":
                    replacement = f"[{pii_type.upper()}_REDACTED]"
                else:
                    replacement = f"[{pii_type.upper()}_ANONYMIZED]"

                anonymized_text = anonymized_text.replace(match, replacement)

        return anonymized_text

    def generate_pseudonym(self, data: str) -> str:
        """Generate consistent pseudonym for data."""
        hash_value = self.hash_data(data)
        # Create readable pseudonym from hash
        return f"USER_{hash_value[:8].upper()}"


class ComplianceManager:
    """
    Manages compliance with data protection regulations.
    Handles GDPR, CCPA, PDPA and other privacy frameworks.
    """

    def __init__(
        self,
        compliance_regions: List[ComplianceRegion] = None,
        data_controller: str = "Quantum Task Planner",
        data_processor: str = "Quantum Task Planner",
    ):

        self.compliance_regions = compliance_regions or [ComplianceRegion.EU_GDPR]
        self.data_controller = data_controller
        self.data_processor = data_processor

        # Components
        self.pii_detector = PIIDetector()
        self.anonymizer = DataAnonymizer()

        # Configuration
        self.pii_fields: Dict[str, PIIField] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}

        # Setup default PII field configurations
        self._setup_default_pii_config()

    def _setup_default_pii_config(self) -> None:
        """Setup default PII field configurations."""
        default_fields = [
            PIIField(
                "user_id", DataClassification.PII, DataRetentionPolicy.MEDIUM_TERM
            ),
            PIIField("email", DataClassification.PII, DataRetentionPolicy.MEDIUM_TERM),
            PIIField(
                "ip_address", DataClassification.PII, DataRetentionPolicy.SHORT_TERM
            ),
            PIIField(
                "user_agent",
                DataClassification.INTERNAL,
                DataRetentionPolicy.SHORT_TERM,
            ),
            PIIField(
                "session_id",
                DataClassification.INTERNAL,
                DataRetentionPolicy.SHORT_TERM,
            ),
            PIIField(
                "task_description",
                DataClassification.INTERNAL,
                DataRetentionPolicy.MEDIUM_TERM,
            ),
            PIIField(
                "error_message",
                DataClassification.INTERNAL,
                DataRetentionPolicy.SHORT_TERM,
            ),
        ]

        for field in default_fields:
            self.pii_fields[field.field_name] = field

    def add_pii_field(self, field: PIIField) -> None:
        """Add or update PII field configuration."""
        self.pii_fields[field.field_name] = field
        logger.info(f"Added PII field configuration: {field.field_name}")

    def classify_data(self, field_name: str, value: Any) -> DataClassification:
        """Classify data based on field configuration and content."""
        # Check configured field
        if field_name in self.pii_fields:
            return self.pii_fields[field_name].classification

        # Auto-detect based on content
        if isinstance(value, str):
            return self.pii_detector.classify_text(value)

        return DataClassification.INTERNAL

    def record_processing_activity(
        self,
        processing_purpose: str,
        data_categories: List[DataClassification],
        legal_basis: str,
        data_subject_id: Optional[str] = None,
        automated_decision_making: bool = False,
        profiling: bool = False,
    ) -> str:
        """Record a data processing activity."""

        record = DataProcessingRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data_subject_id=data_subject_id,
            processing_purpose=processing_purpose,
            data_categories=data_categories,
            legal_basis=legal_basis,
            retention_period=self._calculate_retention_period(data_categories),
            data_controller=self.data_controller,
            data_processor=self.data_processor,
            automated_decision_making=automated_decision_making,
            profiling=profiling,
        )

        self.processing_records.append(record)
        logger.info(f"Recorded processing activity: {record.record_id}")

        return record.record_id

    def _calculate_retention_period(
        self, data_categories: List[DataClassification]
    ) -> int:
        """Calculate retention period based on data categories."""
        max_retention = 30  # Default

        for category in data_categories:
            if category in [DataClassification.RESTRICTED, DataClassification.PII]:
                max_retention = max(max_retention, 365)  # 1 year for PII
            elif category == DataClassification.SENSITIVE:
                max_retention = max(max_retention, 90)  # 3 months for sensitive

        return max_retention

    def record_consent(
        self,
        data_subject_id: str,
        purposes: List[str],
        consent_given: bool,
        consent_method: str = "explicit",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Record user consent."""

        consent = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            data_subject_id=data_subject_id,
            timestamp=datetime.now(),
            consent_given=consent_given,
            purposes=purposes,
            lawful_basis="consent" if consent_given else "withdrawn",
            consent_method=consent_method,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.consent_records[data_subject_id] = consent
        logger.info(f"Recorded consent for {data_subject_id}: {consent_given}")

        return consent.consent_id

    def withdraw_consent(self, data_subject_id: str) -> bool:
        """Withdraw consent for a data subject."""
        if data_subject_id in self.consent_records:
            consent = self.consent_records[data_subject_id]
            consent.withdrawal_timestamp = datetime.now()
            consent.consent_given = False

            logger.info(f"Consent withdrawn for {data_subject_id}")
            return True

        return False

    def has_valid_consent(self, data_subject_id: str, purpose: str) -> bool:
        """Check if valid consent exists for a specific purpose."""
        if data_subject_id not in self.consent_records:
            return False

        consent = self.consent_records[data_subject_id]
        return consent.is_valid() and purpose in consent.purposes

    def anonymize_data(
        self, data: Dict[str, Any], method: str = "hash"
    ) -> Dict[str, Any]:
        """Anonymize data according to field configurations."""
        anonymized = {}

        for field_name, value in data.items():
            classification = self.classify_data(field_name, value)

            if classification in [
                DataClassification.PII,
                DataClassification.RESTRICTED,
                DataClassification.SENSITIVE,
            ]:
                if isinstance(value, str):
                    anonymized[field_name] = self.anonymizer.anonymize_text(
                        value, method
                    )
                else:
                    anonymized[field_name] = self.anonymizer.hash_data(str(value))
            else:
                anonymized[field_name] = value

        return anonymized

    def get_data_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get all data for a specific data subject (GDPR Article 15)."""
        subject_data = {
            "data_subject_id": data_subject_id,
            "processing_records": [],
            "consent_records": [],
            "data_categories": set(),
            "retention_periods": {},
            "lawful_bases": set(),
        }

        # Find processing records
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                subject_data["processing_records"].append(record.to_dict())
                subject_data["data_categories"].update(
                    [cat.value for cat in record.data_categories]
                )
                subject_data["lawful_bases"].add(record.legal_basis)

        # Find consent records
        if data_subject_id in self.consent_records:
            consent = self.consent_records[data_subject_id]
            subject_data["consent_records"].append(
                {
                    "consent_id": consent.consent_id,
                    "timestamp": consent.timestamp.isoformat(),
                    "consent_given": consent.consent_given,
                    "purposes": consent.purposes,
                    "withdrawal_timestamp": (
                        consent.withdrawal_timestamp.isoformat()
                        if consent.withdrawal_timestamp
                        else None
                    ),
                }
            )

        # Convert sets to lists for JSON serialization
        subject_data["data_categories"] = list(subject_data["data_categories"])
        subject_data["lawful_bases"] = list(subject_data["lawful_bases"])

        return subject_data

    def delete_data_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Delete all data for a specific data subject (GDPR Article 17)."""
        deleted_data = {
            "data_subject_id": data_subject_id,
            "deleted_processing_records": 0,
            "deleted_consent_records": 0,
            "deletion_timestamp": datetime.now().isoformat(),
        }

        # Delete processing records
        original_count = len(self.processing_records)
        self.processing_records = [
            r for r in self.processing_records if r.data_subject_id != data_subject_id
        ]
        deleted_data["deleted_processing_records"] = original_count - len(
            self.processing_records
        )

        # Delete consent records
        if data_subject_id in self.consent_records:
            del self.consent_records[data_subject_id]
            deleted_data["deleted_consent_records"] = 1

        logger.info(f"Deleted all data for data subject: {data_subject_id}")
        return deleted_data

    def generate_privacy_notice(self) -> str:
        """Generate privacy notice/policy text."""
        notice = f"""
PRIVACY NOTICE - QUANTUM TASK PLANNER

Data Controller: {self.data_controller}
Data Processor: {self.data_processor}

1. DATA WE COLLECT
We collect and process the following categories of personal data:
- Task execution data (for performance optimization)
- System usage metrics (for monitoring and improvement)
- Error logs (for debugging and system reliability)

2. LAWFUL BASIS FOR PROCESSING
We process your personal data based on:
- Legitimate interests (system optimization and security)
- Consent (where explicitly given)
- Contract performance (service delivery)

3. DATA RETENTION
We retain personal data for the following periods:
- Task execution data: Up to 1 year
- System metrics: Up to 3 months
- Error logs: Up to 30 days

4. YOUR RIGHTS
Under applicable data protection laws, you have the right to:
- Access your personal data
- Rectify inaccurate data
- Erase your data (right to be forgotten)
- Restrict processing
- Data portability
- Object to processing

5. CONTACT INFORMATION
To exercise your rights or for privacy-related questions, contact:
Data Protection Officer: {self.data_controller}

Last updated: {datetime.now().strftime('%Y-%m-%d')}
        """

        return notice.strip()

    def perform_privacy_impact_assessment(self) -> Dict[str, Any]:
        """Perform a basic Privacy Impact Assessment (PIA)."""
        assessment = {
            "assessment_date": datetime.now().isoformat(),
            "data_categories_processed": [],
            "risk_level": "low",
            "risks_identified": [],
            "mitigation_measures": [],
            "compliance_status": {},
        }

        # Analyze data categories
        all_categories = set()
        for record in self.processing_records:
            all_categories.update(record.data_categories)

        assessment["data_categories_processed"] = [cat.value for cat in all_categories]

        # Assess risk level
        if DataClassification.RESTRICTED in all_categories:
            assessment["risk_level"] = "high"
            assessment["risks_identified"].append(
                "Processing of restricted/sensitive personal data"
            )
        elif DataClassification.PII in all_categories:
            assessment["risk_level"] = "medium"
            assessment["risks_identified"].append(
                "Processing of personally identifiable information"
            )

        # Check for automated decision making
        automated_decisions = any(
            r.automated_decision_making for r in self.processing_records
        )
        if automated_decisions:
            assessment["risks_identified"].append("Automated decision making processes")
            assessment["risk_level"] = (
                "medium" if assessment["risk_level"] == "low" else "high"
            )

        # Mitigation measures
        assessment["mitigation_measures"] = [
            "Data minimization principles applied",
            "Anonymization and pseudonymization where possible",
            "Access controls and encryption in place",
            "Regular data retention review and cleanup",
            "User consent management system",
            "Data subject rights fulfillment process",
        ]

        # Compliance status
        for region in self.compliance_regions:
            assessment["compliance_status"][region.value] = {
                "compliant": True,
                "notes": "Basic compliance measures implemented",
            }

        return assessment

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "report_date": datetime.now().isoformat(),
            "compliance_regions": [region.value for region in self.compliance_regions],
            "data_controller": self.data_controller,
            "data_processor": self.data_processor,
            "statistics": {
                "total_processing_records": len(self.processing_records),
                "total_consent_records": len(self.consent_records),
                "active_consents": sum(
                    1 for c in self.consent_records.values() if c.is_valid()
                ),
                "pii_fields_configured": len(self.pii_fields),
            },
            "privacy_impact_assessment": self.perform_privacy_impact_assessment(),
            "data_retention_summary": self._get_retention_summary(),
            "recommendations": self._get_compliance_recommendations(),
        }

        return report

    def _get_retention_summary(self) -> Dict[str, Any]:
        """Get summary of data retention policies."""
        retention_summary = {}

        for field_name, field_config in self.pii_fields.items():
            retention_days = field_config.get_retention_days()
            retention_summary[field_name] = {
                "retention_policy": field_config.retention_policy.value,
                "retention_days": (
                    retention_days if retention_days >= 0 else "permanent"
                ),
                "classification": field_config.classification.value,
            }

        return retention_summary

    def _get_compliance_recommendations(self) -> List[str]:
        """Get compliance recommendations."""
        recommendations = [
            "Regularly review and update privacy policies",
            "Conduct periodic privacy impact assessments",
            "Implement data breach notification procedures",
            "Provide privacy training for staff",
            "Establish data subject request fulfillment processes",
            "Review third-party data sharing agreements",
            "Implement privacy by design principles",
            "Regular audit of data processing activities",
        ]

        return recommendations


# Global compliance manager instance
compliance_manager = ComplianceManager()


# Convenience functions
def record_processing(
    purpose: str, data_categories: List[DataClassification], legal_basis: str, **kwargs
) -> str:
    """Global function to record data processing."""
    return compliance_manager.record_processing_activity(
        purpose, data_categories, legal_basis, **kwargs
    )


def anonymize_data(data: Dict[str, Any], method: str = "hash") -> Dict[str, Any]:
    """Global function to anonymize data."""
    return compliance_manager.anonymize_data(data, method)


def has_consent(data_subject_id: str, purpose: str) -> bool:
    """Global function to check consent."""
    return compliance_manager.has_valid_consent(data_subject_id, purpose)
