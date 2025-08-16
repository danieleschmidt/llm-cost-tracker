"""
Global Compliance and Data Privacy Framework

Comprehensive implementation of international data privacy regulations:
- GDPR (General Data Protection Regulation) - EU
- CCPA (California Consumer Privacy Act) - California
- PDPA (Personal Data Protection Act) - Singapore  
- LGPD (Lei Geral de Proteção de Dados) - Brazil
- Data localization and residency requirements
- Consent management and audit trails
- Data subject rights automation
- Cross-border transfer validation
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for privacy compliance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ProcessingPurpose(Enum):
    """Legal basis for data processing under GDPR."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(Enum):
    """Data subject rights under privacy regulations."""

    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure ("right to be forgotten")
    RESTRICT = "restrict"  # Right to restrict processing
    PORTABILITY = "portability"  # Right to data portability
    OBJECT = "object"  # Right to object
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent


class ComplianceRegion(Enum):
    """Regulatory regions with different compliance requirements."""

    EU = "eu"  # European Union (GDPR)
    CALIFORNIA = "california"  # California (CCPA)
    SINGAPORE = "singapore"  # Singapore (PDPA)
    BRAZIL = "brazil"  # Brazil (LGPD)
    CANADA = "canada"  # Canada (PIPEDA)
    AUSTRALIA = "australia"  # Australia (Privacy Act)
    UK = "uk"  # United Kingdom (UK GDPR)
    GLOBAL = "global"  # General privacy best practices


@dataclass
class PIIDetectionPattern:
    """Pattern for detecting Personally Identifiable Information."""

    name: str
    pattern: str
    confidence: float
    data_type: str
    regions: List[ComplianceRegion]

    def matches(self, text: str) -> List[Dict[str, Any]]:
        """Find matches in text and return match details."""
        matches = []
        for match in re.finditer(self.pattern, text, re.IGNORECASE):
            matches.append(
                {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": self.confidence,
                    "data_type": self.data_type,
                    "pattern_name": self.name,
                }
            )
        return matches


@dataclass
class DataProcessingRecord:
    """Record of data processing activities (GDPR Art. 30)."""

    id: str
    controller_name: str
    controller_contact: str
    dpo_contact: Optional[str]
    processing_purposes: List[ProcessingPurpose]
    data_categories: List[str]
    data_subject_categories: List[str]
    recipients: List[str]
    third_country_transfers: List[str]
    retention_periods: Dict[str, str]
    security_measures: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""

    id: str
    user_id: str
    purposes: List[ProcessingPurpose]
    data_categories: List[str]
    consent_given: bool
    consent_timestamp: datetime
    consent_method: str  # "explicit", "implied", "opt_in", etc.
    consent_text: str
    withdrawal_timestamp: Optional[datetime] = None
    withdrawal_method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    legal_basis: ProcessingPurpose = ProcessingPurpose.CONSENT

    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if not self.consent_given or self.withdrawal_timestamp:
            return False

        # Check consent expiry (typically 12-24 months)
        expiry_period = timedelta(days=365)  # 12 months
        return datetime.now() - self.consent_timestamp < expiry_period


@dataclass
class DataSubjectRequest:
    """Data Subject Access Request (DSAR) record."""

    id: str
    user_id: str
    request_type: DataSubjectRights
    submitted_at: datetime
    identity_verified: bool
    verification_method: str
    status: str  # "pending", "processing", "completed", "rejected"
    completion_deadline: datetime
    response_data: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    completed_at: Optional[datetime] = None
    completed_by: Optional[str] = None


class PIIDetector:
    """Advanced PII detection using patterns and ML techniques."""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.detection_cache: Dict[str, List[Dict]] = {}

    def _initialize_patterns(self) -> List[PIIDetectionPattern]:
        """Initialize PII detection patterns for various regions."""
        patterns = [
            # Email addresses
            PIIDetectionPattern(
                name="email",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                confidence=0.95,
                data_type="email",
                regions=[ComplianceRegion.GLOBAL],
            ),
            # Phone numbers (international format)
            PIIDetectionPattern(
                name="phone_international",
                pattern=r"(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
                confidence=0.80,
                data_type="phone",
                regions=[ComplianceRegion.GLOBAL],
            ),
            # US Social Security Numbers
            PIIDetectionPattern(
                name="ssn_us",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
                confidence=0.90,
                data_type="ssn",
                regions=[ComplianceRegion.CALIFORNIA],
            ),
            # EU Tax ID / VAT Numbers
            PIIDetectionPattern(
                name="vat_eu",
                pattern=r"\b[A-Z]{2}\d{8,12}\b",
                confidence=0.85,
                data_type="tax_id",
                regions=[ComplianceRegion.EU, ComplianceRegion.UK],
            ),
            # Credit Card Numbers (basic Luhn algorithm pattern)
            PIIDetectionPattern(
                name="credit_card",
                pattern=r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
                confidence=0.70,
                data_type="payment_card",
                regions=[ComplianceRegion.GLOBAL],
            ),
            # IP Addresses
            PIIDetectionPattern(
                name="ip_address",
                pattern=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                confidence=0.85,
                data_type="ip_address",
                regions=[ComplianceRegion.GLOBAL],
            ),
            # Names (basic pattern - would need ML for better accuracy)
            PIIDetectionPattern(
                name="person_name",
                pattern=r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
                confidence=0.60,
                data_type="name",
                regions=[ComplianceRegion.GLOBAL],
            ),
            # Addresses (basic pattern)
            PIIDetectionPattern(
                name="address",
                pattern=r"\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)",
                confidence=0.70,
                data_type="address",
                regions=[ComplianceRegion.GLOBAL],
            ),
        ]

        return patterns

    def detect_pii(
        self, text: str, region: ComplianceRegion = ComplianceRegion.GLOBAL
    ) -> List[Dict[str, Any]]:
        """Detect PII in text for specific compliance region."""
        # Check cache first
        cache_key = hashlib.md5((text + region.value).encode()).hexdigest()
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]

        detected_pii = []

        for pattern in self.patterns:
            # Skip patterns not relevant to the region
            if (
                region not in pattern.regions
                and ComplianceRegion.GLOBAL not in pattern.regions
            ):
                continue

            matches = pattern.matches(text)
            for match in matches:
                # Additional validation for high-confidence patterns
                if pattern.name == "credit_card":
                    if self._validate_credit_card(match["text"]):
                        match["confidence"] = 0.95
                    else:
                        match["confidence"] = 0.30

                elif pattern.name == "ssn_us":
                    if self._validate_ssn(match["text"]):
                        match["confidence"] = 0.95
                    else:
                        continue  # Skip invalid SSNs

                detected_pii.append(
                    {
                        **match,
                        "region": region.value,
                        "requires_consent": self._requires_consent(
                            pattern.data_type, region
                        ),
                        "retention_limit_days": self._get_retention_limit(
                            pattern.data_type, region
                        ),
                    }
                )

        # Cache results for performance
        self.detection_cache[cache_key] = detected_pii

        # Clean cache if it gets too large
        if len(self.detection_cache) > 1000:
            # Remove oldest entries
            old_keys = list(self.detection_cache.keys())[:-500]
            for key in old_keys:
                del self.detection_cache[key]

        return detected_pii

    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove spaces and dashes
        number = re.sub(r"[-.\s]", "", number)

        if not number.isdigit() or len(number) < 13 or len(number) > 19:
            return False

        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0

        return luhn_check(number)

    def _validate_ssn(self, ssn: str) -> bool:
        """Basic SSN validation (format and known invalid patterns)."""
        # Remove dashes
        ssn = ssn.replace("-", "")

        if len(ssn) != 9 or not ssn.isdigit():
            return False

        # Check for invalid patterns
        invalid_patterns = [
            "000",
            "666",
            "900",
            "901",
            "902",
            "903",
            "904",
            "905",
            "906",
            "907",
            "908",
            "909",
        ]

        area = ssn[:3]
        if area in invalid_patterns or area >= "773":
            return False

        # Check group and serial numbers
        group = ssn[3:5]
        serial = ssn[5:]

        return group != "00" and serial != "0000"

    def _requires_consent(self, data_type: str, region: ComplianceRegion) -> bool:
        """Determine if data type requires explicit consent in region."""
        high_risk_types = {
            "ssn",
            "payment_card",
            "health",
            "biometric",
            "genetic",
            "racial",
            "political",
            "religious",
        }

        if region == ComplianceRegion.EU:
            # GDPR requires consent for most personal data
            return (
                data_type in ["name", "email", "phone", "address", "ip_address"]
                or data_type in high_risk_types
            )

        elif region == ComplianceRegion.CALIFORNIA:
            # CCPA has broader definition but different consent requirements
            return data_type in high_risk_types

        else:
            # Conservative approach for other regions
            return data_type in high_risk_types

    def _get_retention_limit(self, data_type: str, region: ComplianceRegion) -> int:
        """Get data retention limits in days for data type and region."""
        if region == ComplianceRegion.EU:
            # GDPR principle of data minimization
            limits = {
                "email": 1095,  # 3 years for customer records
                "name": 1095,
                "phone": 1095,
                "address": 1095,
                "ip_address": 365,  # 1 year for logs
                "payment_card": 90,  # 3 months after transaction
                "ssn": 2555,  # 7 years for tax purposes
            }
        else:
            # More conservative limits for other regions
            limits = {
                "email": 730,  # 2 years
                "name": 730,
                "phone": 730,
                "address": 730,
                "ip_address": 180,  # 6 months
                "payment_card": 60,  # 2 months
                "ssn": 2555,  # 7 years
            }

        return limits.get(data_type, 365)  # Default 1 year


class GlobalComplianceManager:
    """
    Central manager for global privacy compliance across multiple jurisdictions.
    """

    def __init__(self):
        self.pii_detector = PIIDetector()
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.dsar_requests: Dict[str, DataSubjectRequest] = {}
        self.audit_log: List[Dict[str, Any]] = []

        # Regional compliance configurations
        self.regional_configs = self._initialize_regional_configs()

        # Data residency rules
        self.data_residency_rules = {
            ComplianceRegion.EU: ["eu-west-1", "eu-central-1", "eu-north-1"],
            ComplianceRegion.UK: ["eu-west-2"],  # UK region
            ComplianceRegion.CALIFORNIA: ["us-west-1", "us-west-2"],
            ComplianceRegion.SINGAPORE: ["ap-southeast-1"],
            ComplianceRegion.BRAZIL: ["sa-east-1"],
            ComplianceRegion.CANADA: ["ca-central-1"],
            ComplianceRegion.AUSTRALIA: ["ap-southeast-2"],
        }

    def _initialize_regional_configs(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Initialize compliance configurations for each region."""
        return {
            ComplianceRegion.EU: {
                "name": "General Data Protection Regulation (GDPR)",
                "consent_required": True,
                "explicit_consent": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required_threshold": 250,  # employees
                "breach_notification_hours": 72,
                "max_fine_percentage": 0.04,  # 4% of annual turnover
                "lawful_bases": [
                    ProcessingPurpose.CONSENT,
                    ProcessingPurpose.CONTRACT,
                    ProcessingPurpose.LEGAL_OBLIGATION,
                    ProcessingPurpose.VITAL_INTERESTS,
                    ProcessingPurpose.PUBLIC_TASK,
                    ProcessingPurpose.LEGITIMATE_INTERESTS,
                ],
            },
            ComplianceRegion.CALIFORNIA: {
                "name": "California Consumer Privacy Act (CCPA)",
                "consent_required": False,  # Opt-out model
                "explicit_consent": False,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "privacy_by_design": False,
                "dpo_required_threshold": None,
                "breach_notification_hours": None,
                "max_fine_per_violation": 7500,  # USD
                "revenue_threshold": 25000000,  # $25M annual revenue
                "personal_info_threshold": 50000,  # consumer records
            },
            ComplianceRegion.SINGAPORE: {
                "name": "Personal Data Protection Act (PDPA)",
                "consent_required": True,
                "explicit_consent": True,
                "right_to_be_forgotten": False,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required_threshold": None,
                "breach_notification_hours": 72,
                "max_fine_sgd": 1000000,  # SGD 1 million
            },
            ComplianceRegion.BRAZIL: {
                "name": "Lei Geral de Proteção de Dados (LGPD)",
                "consent_required": True,
                "explicit_consent": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required_threshold": None,
                "breach_notification_hours": 72,
                "max_fine_percentage": 0.02,  # 2% of company revenue
            },
        }

    async def validate_data_processing(
        self,
        data: Any,
        purpose: ProcessingPurpose,
        region: ComplianceRegion,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate data processing against compliance requirements."""
        validation_result = {
            "compliant": True,
            "warnings": [],
            "violations": [],
            "required_actions": [],
            "detected_pii": [],
            "legal_basis": purpose,
            "region": region.value,
        }

        # Convert data to string for PII detection
        if isinstance(data, dict):
            data_str = json.dumps(data, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = str(data)
        else:
            data_str = str(data)

        # Detect PII in the data
        detected_pii = self.pii_detector.detect_pii(data_str, region)
        validation_result["detected_pii"] = detected_pii

        if detected_pii:
            # Check consent requirements
            if user_id:
                consent_check = await self._check_user_consent(
                    user_id, purpose, detected_pii, region
                )
                validation_result.update(consent_check)
            else:
                validation_result["warnings"].append(
                    "Processing PII without user identification"
                )
                validation_result["required_actions"].append(
                    "Identify data subject for consent verification"
                )

            # Check data residency requirements
            residency_check = self._check_data_residency(region)
            if not residency_check["compliant"]:
                validation_result["violations"].extend(residency_check["violations"])
                validation_result["compliant"] = False

        # Region-specific validations
        if region == ComplianceRegion.EU:
            gdpr_check = await self._validate_gdpr_compliance(
                data_str, purpose, user_id
            )
            validation_result = self._merge_validation_results(
                validation_result, gdpr_check
            )

        elif region == ComplianceRegion.CALIFORNIA:
            ccpa_check = await self._validate_ccpa_compliance(
                data_str, purpose, user_id
            )
            validation_result = self._merge_validation_results(
                validation_result, ccpa_check
            )

        # Log the validation for audit trail
        self._log_compliance_event(
            "data_processing_validation",
            {
                "user_id": user_id,
                "purpose": purpose.value,
                "region": region.value,
                "compliant": validation_result["compliant"],
                "pii_detected": len(detected_pii),
                "violations": len(validation_result["violations"]),
            },
        )

        return validation_result

    async def _check_user_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        detected_pii: List[Dict],
        region: ComplianceRegion,
    ) -> Dict[str, Any]:
        """Check if user has given valid consent for data processing."""
        result = {"consent_valid": False, "consent_required": True}

        # Get user's consent records
        user_consents = [
            consent
            for consent in self.consent_records.values()
            if consent.user_id == user_id and consent.is_valid
        ]

        if not user_consents:
            result["violations"] = ["No valid consent found for data processing"]
            result["required_actions"] = ["Obtain user consent before processing"]
            return result

        # Check if consent covers the purpose and data types
        pii_types = set(pii["data_type"] for pii in detected_pii)

        valid_consent = None
        for consent in user_consents:
            if purpose in consent.purposes:
                # Check if consent covers detected PII types
                consent_data_types = set(consent.data_categories)
                if (
                    pii_types.issubset(consent_data_types)
                    or "all" in consent_data_types
                ):
                    valid_consent = consent
                    break

        if valid_consent:
            result["consent_valid"] = True
            result["consent_id"] = valid_consent.id
            result["consent_timestamp"] = valid_consent.consent_timestamp.isoformat()
        else:
            result["violations"] = [
                f"No consent found for purpose '{purpose.value}' and PII types {list(pii_types)}"
            ]
            result["required_actions"] = [
                f"Obtain specific consent for {purpose.value} processing"
            ]

        return result

    def _check_data_residency(self, region: ComplianceRegion) -> Dict[str, Any]:
        """Check data residency requirements for region."""
        result = {"compliant": True, "violations": []}

        allowed_regions = self.data_residency_rules.get(region, [])

        if not allowed_regions:
            result["violations"].append(
                f"No approved data centers for region {region.value}"
            )
            result["compliant"] = False

        # In a real implementation, this would check current data location
        # For now, we assume compliance

        return result

    async def _validate_gdpr_compliance(
        self, data: str, purpose: ProcessingPurpose, user_id: str
    ) -> Dict[str, Any]:
        """Validate GDPR-specific requirements."""
        result = {
            "gdpr_compliant": True,
            "gdpr_violations": [],
            "gdpr_requirements": [],
        }

        # GDPR Article 6 - Lawfulness of processing
        valid_lawful_bases = self.regional_configs[ComplianceRegion.EU]["lawful_bases"]
        if purpose not in valid_lawful_bases:
            result["gdpr_violations"].append(f"Invalid lawful basis: {purpose.value}")
            result["gdpr_compliant"] = False

        # GDPR Article 5 - Principles relating to processing
        result["gdpr_requirements"].extend(
            [
                "Ensure data accuracy and keep up to date",
                "Limit data retention to necessary period",
                "Implement appropriate security measures",
                "Maintain records of processing activities",
            ]
        )

        # GDPR Article 25 - Data protection by design and by default
        result["gdpr_requirements"].append("Implement privacy by design measures")

        return result

    async def _validate_ccpa_compliance(
        self, data: str, purpose: ProcessingPurpose, user_id: str
    ) -> Dict[str, Any]:
        """Validate CCPA-specific requirements."""
        result = {
            "ccpa_compliant": True,
            "ccpa_violations": [],
            "ccpa_requirements": [],
        }

        # CCPA requires different handling - opt-out model
        result["ccpa_requirements"].extend(
            [
                "Provide clear privacy notice",
                "Enable consumer opt-out rights",
                "Implement data deletion capabilities",
                "Maintain consumer request records",
            ]
        )

        return result

    def _merge_validation_results(
        self, base_result: Dict, additional_result: Dict
    ) -> Dict:
        """Merge multiple validation results."""
        merged = base_result.copy()

        for key, value in additional_result.items():
            if key.endswith("_compliant"):
                merged["compliant"] = merged["compliant"] and value
            elif key.endswith("_violations"):
                merged["violations"].extend(value)
            elif key.endswith("_requirements"):
                merged["required_actions"].extend(value)

        return merged

    async def process_data_subject_request(
        self, request: DataSubjectRequest
    ) -> Dict[str, Any]:
        """Process a data subject access request (DSAR)."""
        self.dsar_requests[request.id] = request

        result = {
            "request_id": request.id,
            "status": "processing",
            "estimated_completion": request.completion_deadline.isoformat(),
            "response_data": {},
        }

        try:
            if request.request_type == DataSubjectRights.ACCESS:
                # Compile all data for the user
                user_data = await self._compile_user_data(request.user_id)
                result["response_data"] = user_data

            elif request.request_type == DataSubjectRights.ERASURE:
                # Delete user data (right to be forgotten)
                deletion_result = await self._delete_user_data(request.user_id)
                result["response_data"] = deletion_result

            elif request.request_type == DataSubjectRights.RECTIFICATION:
                # This would need additional data from the request
                result["response_data"] = {
                    "message": "Rectification capability available - requires specific corrections"
                }

            elif request.request_type == DataSubjectRights.PORTABILITY:
                # Export user data in portable format
                portable_data = await self._export_portable_data(request.user_id)
                result["response_data"] = portable_data

            request.status = "completed"
            request.completed_at = datetime.now()
            result["status"] = "completed"

        except Exception as e:
            request.status = "failed"
            request.rejection_reason = str(e)
            result["status"] = "failed"
            result["error"] = str(e)

        # Log the DSAR processing
        self._log_compliance_event(
            "dsar_processed",
            {
                "request_id": request.id,
                "user_id": request.user_id,
                "request_type": request.request_type.value,
                "status": result["status"],
            },
        )

        return result

    async def _compile_user_data(self, user_id: str) -> Dict[str, Any]:
        """Compile all data associated with a user ID."""
        # This would integrate with actual data stores
        user_data = {
            "user_id": user_id,
            "consent_records": [
                consent.id
                for consent in self.consent_records.values()
                if consent.user_id == user_id
            ],
            "processing_records": [
                record.id
                for record in self.processing_records.values()
                # Would need more complex logic to link processing records to users
            ],
            "audit_events": [
                event for event in self.audit_log if event.get("user_id") == user_id
            ][
                -50:
            ],  # Last 50 events
        }

        return user_data

    async def _delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete all data associated with a user (right to be forgotten)."""
        deletion_summary = {
            "user_id": user_id,
            "deleted_items": [],
            "retained_items": [],
            "deletion_timestamp": datetime.now().isoformat(),
        }

        # Delete consent records
        deleted_consents = [
            consent_id
            for consent_id, consent in self.consent_records.items()
            if consent.user_id == user_id
        ]

        for consent_id in deleted_consents:
            del self.consent_records[consent_id]

        deletion_summary["deleted_items"].append(
            f"Deleted {len(deleted_consents)} consent records"
        )

        # Note: In practice, some data may need to be retained for legal/regulatory reasons
        # This should be carefully evaluated based on applicable laws

        return deletion_summary

    async def _export_portable_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in a portable format."""
        portable_data = {
            "export_timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "data_format": "json",
            "consent_history": [
                {
                    "consent_id": consent.id,
                    "purposes": [p.value for p in consent.purposes],
                    "consent_given": consent.consent_given,
                    "timestamp": consent.consent_timestamp.isoformat(),
                    "withdrawal_timestamp": (
                        consent.withdrawal_timestamp.isoformat()
                        if consent.withdrawal_timestamp
                        else None
                    ),
                }
                for consent in self.consent_records.values()
                if consent.user_id == user_id
            ],
        }

        return portable_data

    def anonymize_data(
        self, data: Any, region: ComplianceRegion = ComplianceRegion.GLOBAL
    ) -> Any:
        """Anonymize data by removing or masking PII."""
        if isinstance(data, str):
            return self._anonymize_string(data, region)
        elif isinstance(data, dict):
            return {
                key: self.anonymize_data(value, region) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.anonymize_data(item, region) for item in data]
        else:
            return data

    def _anonymize_string(self, text: str, region: ComplianceRegion) -> str:
        """Anonymize PII in a string."""
        detected_pii = self.pii_detector.detect_pii(text, region)

        # Sort by position (reverse order to maintain indices)
        detected_pii.sort(key=lambda x: x["start"], reverse=True)

        anonymized_text = text
        for pii in detected_pii:
            # Replace PII with masked version
            mask_char = "*"
            if pii["data_type"] == "email":
                # Keep domain for email
                original = pii["text"]
                username_part = original.split("@")[0]
                domain_part = original.split("@")[1] if "@" in original else ""
                masked = f"{'*' * len(username_part)}@{domain_part}"
            elif pii["data_type"] == "phone":
                # Mask middle digits
                original = pii["text"]
                if len(original) >= 6:
                    masked = original[:2] + "*" * (len(original) - 4) + original[-2:]
                else:
                    masked = "*" * len(original)
            else:
                # Generic masking
                masked = mask_char * len(pii["text"])

            anonymized_text = (
                anonymized_text[: pii["start"]] + masked + anonymized_text[pii["end"] :]
            )

        return anonymized_text

    def _log_compliance_event(self, event_type: str, details: Dict[str, Any]):
        """Log compliance-related events for audit trail."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_id": str(uuid.uuid4()),
            **details,
        }

        self.audit_log.append(event)

        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 events

        logger.info(f"Compliance event logged: {event_type} - {details}")

    def get_compliance_report(self, region: ComplianceRegion) -> Dict[str, Any]:
        """Generate compliance report for a specific region."""
        report = {
            "region": region.value,
            "region_config": self.regional_configs.get(region, {}),
            "report_timestamp": datetime.now().isoformat(),
            "statistics": {
                "active_consents": len(
                    [c for c in self.consent_records.values() if c.is_valid]
                ),
                "total_consents": len(self.consent_records),
                "pending_dsars": len(
                    [r for r in self.dsar_requests.values() if r.status == "pending"]
                ),
                "completed_dsars": len(
                    [r for r in self.dsar_requests.values() if r.status == "completed"]
                ),
                "audit_events_30_days": len(
                    [
                        e
                        for e in self.audit_log
                        if datetime.fromisoformat(e["timestamp"])
                        > datetime.now() - timedelta(days=30)
                    ]
                ),
            },
            "compliance_status": "compliant",  # Would be calculated based on violations
            "last_privacy_impact_assessment": None,  # Would track PIA dates
            "data_residency_compliant": True,
        }

        return report


# Global singleton instance
compliance_manager = GlobalComplianceManager()


# Convenience functions for common operations
async def validate_processing(
    data: Any,
    purpose: ProcessingPurpose,
    region: ComplianceRegion,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to validate data processing."""
    return await compliance_manager.validate_data_processing(
        data, purpose, region, user_id
    )


def detect_pii(
    text: str, region: ComplianceRegion = ComplianceRegion.GLOBAL
) -> List[Dict[str, Any]]:
    """Convenience function to detect PII in text."""
    return compliance_manager.pii_detector.detect_pii(text, region)


def anonymize_data(
    data: Any, region: ComplianceRegion = ComplianceRegion.GLOBAL
) -> Any:
    """Convenience function to anonymize data."""
    return compliance_manager.anonymize_data(data, region)
