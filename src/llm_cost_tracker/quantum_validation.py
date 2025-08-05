"""
Comprehensive validation and error handling for Quantum Task Planner.
Provides input validation, security checks, and error recovery mechanisms.
"""

import re
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    
    is_valid: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: List[ValidationResult]
    errors: List[ValidationResult]
    critical_errors: List[ValidationResult]
    
    def get_all_issues(self) -> List[ValidationResult]:
        """Get all validation issues sorted by severity."""
        all_issues = self.critical_errors + self.errors + self.warnings
        return sorted(all_issues, key=lambda x: x.severity.value)


class QuantumTaskValidator:
    """
    Comprehensive validator for quantum tasks and planning operations.
    Provides security checks, business rule validation, and data integrity checks.
    """
    
    def __init__(self):
        # Validation rules configuration
        self.max_task_name_length = 200
        self.max_description_length = 2000
        self.max_task_id_length = 100
        self.max_dependencies = 50
        self.max_resources_per_task = 20
        self.max_priority = 10.0
        self.min_priority = 1.0
        self.max_duration_hours = 24 * 7  # 1 week
        self.max_entangled_tasks = 10
        
        # Security patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'on\w+\s*=',                 # Event handlers
            r'eval\s*\(',                 # Eval calls
            r'exec\s*\(',                 # Exec calls
            r'system\s*\(',               # System calls
            r'subprocess\.',              # Subprocess calls
            r'__import__\s*\(',           # Import calls
            r'file:///',                  # File URLs
            r'ftp://',                    # FTP URLs
        ]
        
        # Reserved task IDs that cannot be used
        self.reserved_ids = {
            'system', 'admin', 'root', 'quantum', 'planner',
            'monitor', 'health', 'metrics', 'config', 'api'
        }
        
        # Valid resource types
        self.valid_resource_types = {
            'cpu_cores', 'memory_gb', 'storage_gb', 'network_bandwidth',
            'gpu_cores', 'disk_iops', 'network_connections'
        }
    
    def validate_task_id(self, task_id: str) -> ValidationResult:
        """Validate task ID format and security."""
        if not task_id:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Task ID cannot be empty",
                field="id",
                error_code="EMPTY_TASK_ID"
            )
        
        if len(task_id) > self.max_task_id_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task ID too long (max {self.max_task_id_length} characters)",
                field="id",
                suggested_fix=f"Shorten ID to {self.max_task_id_length} characters or less",
                error_code="TASK_ID_TOO_LONG"
            )
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, task_id, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Task ID contains potentially dangerous content",
                    field="id",
                    error_code="SECURITY_VIOLATION"
                )
        
        # Check for reserved IDs
        if task_id.lower() in self.reserved_ids:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task ID '{task_id}' is reserved and cannot be used",
                field="id",
                suggested_fix="Choose a different task ID",
                error_code="RESERVED_TASK_ID"
            )
        
        # Check format (alphanumeric, underscore, hyphen only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Task ID can only contain letters, numbers, underscores, and hyphens",
                field="id",
                suggested_fix="Remove special characters from task ID",
                error_code="INVALID_TASK_ID_FORMAT"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Task ID is valid",
            field="id"
        )
    
    def validate_task_name(self, name: str) -> ValidationResult:
        """Validate task name."""
        if not name or not name.strip():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Task name cannot be empty",
                field="name",
                error_code="EMPTY_TASK_NAME"
            )
        
        if len(name) > self.max_task_name_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task name too long (max {self.max_task_name_length} characters)",
                field="name",
                suggested_fix=f"Shorten name to {self.max_task_name_length} characters or less",
                error_code="TASK_NAME_TOO_LONG"
            )
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Task name contains potentially dangerous content",
                    field="name",
                    error_code="SECURITY_VIOLATION"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Task name is valid",
            field="name"
        )
    
    def validate_task_description(self, description: str) -> ValidationResult:
        """Validate task description."""
        if not description or not description.strip():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Task description is empty",
                field="description",
                suggested_fix="Add a meaningful description",
                error_code="EMPTY_DESCRIPTION"
            )
        
        if len(description) > self.max_description_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Description too long (max {self.max_description_length} characters)",
                field="description",
                suggested_fix=f"Shorten description to {self.max_description_length} characters or less",
                error_code="DESCRIPTION_TOO_LONG"
            )
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Description contains potentially dangerous content",
                    field="description",
                    error_code="SECURITY_VIOLATION"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Task description is valid",
            field="description"
        )
    
    def validate_priority(self, priority: float) -> ValidationResult:
        """Validate task priority."""
        if not isinstance(priority, (int, float)):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Priority must be a number",
                field="priority",
                error_code="INVALID_PRIORITY_TYPE"
            )
        
        if priority < self.min_priority or priority > self.max_priority:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Priority must be between {self.min_priority} and {self.max_priority}",
                field="priority",
                suggested_fix=f"Set priority between {self.min_priority} and {self.max_priority}",
                error_code="PRIORITY_OUT_OF_RANGE"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Priority is valid",
            field="priority"
        )
    
    def validate_duration(self, duration: timedelta) -> ValidationResult:
        """Validate task duration."""
        if not isinstance(duration, timedelta):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Duration must be a timedelta object",
                field="estimated_duration",
                error_code="INVALID_DURATION_TYPE"
            )
        
        if duration.total_seconds() <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Duration must be positive",
                field="estimated_duration",
                suggested_fix="Set a positive duration",
                error_code="NEGATIVE_DURATION"
            )
        
        max_duration = timedelta(hours=self.max_duration_hours)
        if duration > max_duration:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Duration exceeds recommended maximum of {self.max_duration_hours} hours",
                field="estimated_duration",
                suggested_fix=f"Consider breaking into smaller tasks or reducing duration",
                error_code="EXCESSIVE_DURATION"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Duration is valid",
            field="estimated_duration"
        )
    
    def validate_resources(self, resources: Dict[str, float]) -> ValidationResult:
        """Validate resource requirements."""
        if not isinstance(resources, dict):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Resources must be a dictionary",
                field="required_resources",
                error_code="INVALID_RESOURCES_TYPE"
            )
        
        if len(resources) > self.max_resources_per_task:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Too many resource types (max {self.max_resources_per_task})",
                field="required_resources",
                suggested_fix=f"Reduce to {self.max_resources_per_task} resource types or less",
                error_code="TOO_MANY_RESOURCES"
            )
        
        for resource_type, amount in resources.items():
            # Validate resource type
            if resource_type not in self.valid_resource_types:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid resource type: {resource_type}",
                    field="required_resources",
                    suggested_fix=f"Use one of: {', '.join(self.valid_resource_types)}",
                    error_code="INVALID_RESOURCE_TYPE"
                )
            
            # Validate resource amount
            if not isinstance(amount, (int, float)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resource amount for {resource_type} must be a number",
                    field="required_resources",
                    error_code="INVALID_RESOURCE_AMOUNT"
                )
            
            if amount < 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resource amount for {resource_type} cannot be negative",
                    field="required_resources",
                    error_code="NEGATIVE_RESOURCE_AMOUNT"
                )
            
            # Reasonable limits check
            resource_limits = {
                'cpu_cores': 128.0,
                'memory_gb': 1024.0,
                'storage_gb': 10000.0,
                'network_bandwidth': 10000.0,  # 10 Gbps
                'gpu_cores': 64.0,
                'disk_iops': 100000.0,
                'network_connections': 10000.0
            }
            
            limit = resource_limits.get(resource_type, float('inf'))
            if amount > limit:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Resource amount for {resource_type} seems excessive: {amount}",
                    field="required_resources",
                    suggested_fix=f"Consider reducing {resource_type} requirement",
                    error_code="EXCESSIVE_RESOURCE_REQUIREMENT"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Resources are valid",
            field="required_resources"
        )
    
    def validate_dependencies(self, dependencies: Set[str], all_task_ids: Set[str]) -> ValidationResult:
        """Validate task dependencies."""
        if not isinstance(dependencies, (set, list)):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Dependencies must be a set or list",
                field="dependencies",
                error_code="INVALID_DEPENDENCIES_TYPE"
            )
        
        dependencies_set = set(dependencies) if isinstance(dependencies, list) else dependencies
        
        if len(dependencies_set) > self.max_dependencies:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Too many dependencies (max {self.max_dependencies})",
                field="dependencies",
                suggested_fix=f"Reduce to {self.max_dependencies} dependencies or less",
                error_code="TOO_MANY_DEPENDENCIES"
            )
        
        # Check for invalid dependency IDs
        for dep_id in dependencies_set:
            if not isinstance(dep_id, str):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Dependency IDs must be strings",
                    field="dependencies",
                    error_code="INVALID_DEPENDENCY_ID_TYPE"
                )
            
            # Validate dependency ID format
            id_validation = self.validate_task_id(dep_id)
            if not id_validation.is_valid:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid dependency ID '{dep_id}': {id_validation.message}",
                    field="dependencies",
                    error_code="INVALID_DEPENDENCY_ID"
                )
        
        # Check for non-existent dependencies (if task list provided)
        if all_task_ids is not None:
            missing_deps = dependencies_set - all_task_ids
            if missing_deps:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Dependencies not found: {', '.join(missing_deps)}",
                    field="dependencies",
                    suggested_fix="Create dependent tasks first or remove invalid dependencies",
                    error_code="MISSING_DEPENDENCIES"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Dependencies are valid",
            field="dependencies"
        )
    
    def detect_circular_dependencies(self, task_id: str, dependencies: Dict[str, Set[str]]) -> ValidationResult:
        """Detect circular dependencies in task graph."""
        
        def has_cycle(current_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            """DFS to detect cycles."""
            visited.add(current_id)
            rec_stack.add(current_id)
            
            for dep_id in dependencies.get(current_id, set()):
                if dep_id not in visited:
                    if has_cycle(dep_id, visited, rec_stack):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(current_id)
            return False
        
        visited = set()
        rec_stack = set()
        
        if has_cycle(task_id, visited, rec_stack):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Circular dependency detected involving task '{task_id}'",
                field="dependencies",
                suggested_fix="Remove circular dependencies to create a valid task graph",
                error_code="CIRCULAR_DEPENDENCY"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="No circular dependencies detected",
            field="dependencies"
        )
    
    def validate_quantum_properties(self, amplitude: complex, interference_pattern: Dict[str, float],
                                  entangled_tasks: Set[str]) -> ValidationResult:
        """Validate quantum-specific properties."""
        
        # Validate probability amplitude
        if not isinstance(amplitude, complex):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Probability amplitude must be a complex number",
                field="probability_amplitude",
                error_code="INVALID_AMPLITUDE_TYPE"
            )
        
        amplitude_magnitude = abs(amplitude)
        if amplitude_magnitude > 1.0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Probability amplitude magnitude ({amplitude_magnitude:.2f}) > 1.0",
                field="probability_amplitude",
                suggested_fix="Normalize amplitude to have magnitude ≤ 1.0",
                error_code="AMPLITUDE_TOO_LARGE"
            )
        
        # Validate interference pattern
        for task_id, effect in interference_pattern.items():
            if not isinstance(effect, (int, float)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Interference effect for {task_id} must be a number",
                    field="interference_pattern",
                    error_code="INVALID_INTERFERENCE_EFFECT"
                )
            
            if abs(effect) > 1.0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Large interference effect for {task_id}: {effect}",
                    field="interference_pattern",
                    suggested_fix="Consider reducing interference effect magnitude",
                    error_code="LARGE_INTERFERENCE_EFFECT"
                )
        
        # Validate entangled tasks
        if len(entangled_tasks) > self.max_entangled_tasks:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Too many entangled tasks ({len(entangled_tasks)}), max recommended: {self.max_entangled_tasks}",
                field="entangled_tasks",
                suggested_fix="Reduce number of entangled tasks for better performance",
                error_code="TOO_MANY_ENTANGLEMENTS"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Quantum properties are valid",
            field="quantum_properties"
        )
    
    def validate_complete_task(self, task_data: Dict[str, Any], 
                             existing_tasks: Optional[Set[str]] = None) -> ValidationSummary:
        """Perform complete validation of a task."""
        
        results = []
        
        # Basic field validations
        results.append(self.validate_task_id(task_data.get('id', '')))
        results.append(self.validate_task_name(task_data.get('name', '')))
        results.append(self.validate_task_description(task_data.get('description', '')))
        results.append(self.validate_priority(task_data.get('priority', 0)))
        
        # Duration validation
        duration = task_data.get('estimated_duration')
        if duration:
            results.append(self.validate_duration(duration))
        
        # Resources validation
        resources = task_data.get('required_resources', {})
        results.append(self.validate_resources(resources))
        
        # Dependencies validation
        dependencies = task_data.get('dependencies', set())
        results.append(self.validate_dependencies(dependencies, existing_tasks))
        
        # Quantum properties validation
        amplitude = task_data.get('probability_amplitude', complex(1.0, 0.0))
        interference = task_data.get('interference_pattern', {})
        entangled = task_data.get('entangled_tasks', set())
        results.append(self.validate_quantum_properties(amplitude, interference, entangled))
        
        # Categorize results
        warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        critical_errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        
        # Overall validation status
        is_valid = len(errors) == 0 and len(critical_errors) == 0
        
        return ValidationSummary(
            is_valid=is_valid,
            total_checks=len(results),
            passed_checks=len([r for r in results if r.is_valid]),
            failed_checks=len([r for r in results if not r.is_valid]),
            warnings=warnings,
            errors=errors,
            critical_errors=critical_errors
        )
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input string by removing dangerous content."""
        if not isinstance(input_str, str):
            return str(input_str)
        
        sanitized = input_str
        
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def suggest_task_id(self, name: str) -> str:
        """Suggest a valid task ID based on the task name."""
        # Convert to lowercase and replace spaces/special chars with underscores
        suggested_id = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        
        # Remove consecutive underscores
        suggested_id = re.sub(r'_+', '_', suggested_id)
        
        # Remove leading/trailing underscores
        suggested_id = suggested_id.strip('_')
        
        # Truncate if too long
        if len(suggested_id) > self.max_task_id_length:
            suggested_id = suggested_id[:self.max_task_id_length]
        
        # Ensure it's not reserved
        if suggested_id.lower() in self.reserved_ids:
            suggested_id = f"task_{suggested_id}"
        
        return suggested_id or "unnamed_task"


# Global validator instance
task_validator = QuantumTaskValidator()


def validate_task_input(task_data: Dict[str, Any], 
                       existing_tasks: Optional[Set[str]] = None) -> ValidationSummary:
    """Convenience function for task validation."""
    return task_validator.validate_complete_task(task_data, existing_tasks)


def sanitize_task_input(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all string inputs in task data."""
    sanitized = task_data.copy()
    
    string_fields = ['id', 'name', 'description']
    for field in string_fields:
        if field in sanitized and isinstance(sanitized[field], str):
            sanitized[field] = task_validator.sanitize_input(sanitized[field])
    
    return sanitized