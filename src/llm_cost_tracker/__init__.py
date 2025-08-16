"""LLM Cost Tracker - OpenTelemetry-based cost tracking for LLM applications with Quantum-Inspired Task Planning."""

__version__ = "0.1.0"

from .quantum_compliance import anonymize_data, compliance_manager, record_processing
from .quantum_i18n import SupportedLanguage, quantum_i18n, set_language, t
from .quantum_task_planner import (
    QuantumTask,
    QuantumTaskPlanner,
    ResourcePool,
    TaskState,
)

__all__ = [
    "QuantumTaskPlanner",
    "QuantumTask",
    "TaskState",
    "ResourcePool",
    "quantum_i18n",
    "t",
    "set_language",
    "SupportedLanguage",
    "compliance_manager",
    "record_processing",
    "anonymize_data",
]
