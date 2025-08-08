"""LLM Cost Tracker - OpenTelemetry-based cost tracking for LLM applications with Quantum-Inspired Task Planning."""

__version__ = "0.1.0"

from .quantum_task_planner import QuantumTaskPlanner, QuantumTask, TaskState, ResourcePool
from .quantum_i18n import quantum_i18n, t, set_language, SupportedLanguage
from .quantum_compliance import compliance_manager, record_processing, anonymize_data

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
    "anonymize_data"
]