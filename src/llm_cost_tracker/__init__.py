"""LLM Cost Tracker — stdlib-only cost tracking and budgeting for LLM APIs."""

from .pricing import ModelPricing, ModelRate
from .tracker import CostTracker, UsageRecord, BudgetGuard, BudgetExceeded
from .report import CostReport

__all__ = [
    "ModelPricing",
    "ModelRate",
    "CostTracker",
    "UsageRecord",
    "BudgetGuard",
    "BudgetExceeded",
    "CostReport",
]

__version__ = "1.0.0"
