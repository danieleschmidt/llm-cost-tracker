"""
Core tracking primitives: UsageRecord, CostTracker, BudgetGuard.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .pricing import ModelPricing


# ── UsageRecord ──────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    """A single LLM API call."""
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tag: str = "default"
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cost_usd: float = 0.0  # filled by CostTracker after insert

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UsageRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def ts(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


# ── BudgetExceeded ────────────────────────────────────────────────────────────

class BudgetExceeded(Exception):
    """Raised when spending exceeds a budget threshold."""
    def __init__(self, tag: str, spent: float, budget: float):
        self.tag = tag
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"Budget exceeded for '{tag}': spent ${spent:.4f} of ${budget:.4f}"
        )


# ── BudgetGuard ───────────────────────────────────────────────────────────────

@dataclass
class BudgetGuard:
    """Monitors spending against a budget threshold."""
    tag: str
    limit_usd: float
    alert_at: float = 0.80   # warn at 80% by default
    raise_on_exceed: bool = True

    def check(self, current_spend: float) -> str | None:
        """
        Check spend against limit.
        Returns None (ok), 'warn', or 'exceeded'.
        Raises BudgetExceeded if raise_on_exceed=True and limit hit.
        """
        ratio = current_spend / self.limit_usd if self.limit_usd > 0 else 0
        if ratio >= 1.0:
            if self.raise_on_exceed:
                raise BudgetExceeded(self.tag, current_spend, self.limit_usd)
            return "exceeded"
        if ratio >= self.alert_at:
            return "warn"
        return None


# ── CostTracker ───────────────────────────────────────────────────────────────

class CostTracker:
    """
    Accumulates UsageRecords, computes costs, persists to a JSON file.

    Storage: ~/.config/llm-cost-tracker/usage.jsonl  (one JSON obj per line)
    Overridable via env LLM_COST_DB or constructor argument.
    """

    DEFAULT_DB = Path.home() / ".config" / "llm-cost-tracker" / "usage.jsonl"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        pricing: Optional[ModelPricing] = None,
        budgets: Optional[list[BudgetGuard]] = None,
    ):
        self.db_path = Path(
            db_path
            or os.environ.get("LLM_COST_DB", self.DEFAULT_DB)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pricing = pricing or ModelPricing()
        self.budgets: dict[str, BudgetGuard] = {b.tag: b for b in (budgets or [])}

    # ── write ────────────────────────────────────────────────────────────────

    def add(self, record: UsageRecord) -> UsageRecord:
        """Compute cost, check budgets, append record to DB."""
        try:
            record.cost_usd = self.pricing.compute_cost(
                record.model, record.input_tokens, record.output_tokens
            )
        except KeyError:
            record.cost_usd = 0.0  # unknown model — log anyway

        with open(self.db_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        # check budget for this tag
        if record.tag in self.budgets:
            spent = sum(r.cost_usd for r in self.iter_records(tag=record.tag))
            self.budgets[record.tag].check(spent)

        return record

    def add_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tag: str = "default",
        timestamp: Optional[str] = None,
    ) -> UsageRecord:
        r = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tag=tag,
        )
        if timestamp:
            r.timestamp = timestamp
        return self.add(r)

    def set_budget(self, guard: BudgetGuard) -> None:
        self.budgets[guard.tag] = guard

    # ── read ─────────────────────────────────────────────────────────────────

    def iter_records(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        tag: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterator[UsageRecord]:
        if not self.db_path.exists():
            return
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    r = UsageRecord.from_dict(d)
                except (json.JSONDecodeError, TypeError):
                    continue
                if since and r.ts < since:
                    continue
                if until and r.ts > until:
                    continue
                if tag and r.tag != tag:
                    continue
                if model and r.model.lower() != model.lower():
                    continue
                yield r

    def records(self, **kwargs) -> list[UsageRecord]:
        return list(self.iter_records(**kwargs))

    def total_cost(self, **kwargs) -> float:
        return sum(r.cost_usd for r in self.iter_records(**kwargs))

    def total_tokens(self, **kwargs) -> dict[str, int]:
        inp = out = 0
        for r in self.iter_records(**kwargs):
            inp += r.input_tokens
            out += r.output_tokens
        return {"input": inp, "output": out, "total": inp + out}
