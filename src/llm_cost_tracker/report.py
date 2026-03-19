"""
CostReport: daily/weekly/monthly breakdowns, per-model and per-tag totals.
Outputs JSON or plain text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from .tracker import CostTracker, UsageRecord


def _bucket(ts: datetime, period: str) -> str:
    """Return a bucket label for a timestamp."""
    if period == "daily":
        return ts.strftime("%Y-%m-%d")
    if period == "weekly":
        monday = ts - timedelta(days=ts.weekday())
        return "week-" + monday.strftime("%Y-%m-%d")
    if period == "monthly":
        return ts.strftime("%Y-%m")
    raise ValueError(f"Unknown period: {period}")


class CostReport:
    """Generate cost reports from a CostTracker."""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    # ── helpers ──────────────────────────────────────────────────────────────

    def _records(self, since: Optional[datetime], until: Optional[datetime]) -> list[UsageRecord]:
        return self.tracker.records(since=since, until=until)

    # ── report data ──────────────────────────────────────────────────────────

    def build(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        period: str = "daily",   # daily | weekly | monthly
    ) -> dict:
        records = self._records(since, until)

        if not records:
            return {
                "period": period,
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "total_cost_usd": 0.0,
                "total_calls": 0,
                "total_tokens": {"input": 0, "output": 0, "total": 0},
                "by_model": {},
                "by_tag": {},
                "by_period": {},
            }

        total_cost = 0.0
        total_input = total_output = 0
        by_model: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0})
        by_tag: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0})
        by_period: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0})

        for r in records:
            total_cost += r.cost_usd
            total_input += r.input_tokens
            total_output += r.output_tokens

            bm = by_model[r.model]
            bm["cost_usd"] += r.cost_usd
            bm["calls"] += 1
            bm["input_tokens"] += r.input_tokens
            bm["output_tokens"] += r.output_tokens

            bt = by_tag[r.tag]
            bt["cost_usd"] += r.cost_usd
            bt["calls"] += 1
            bt["input_tokens"] += r.input_tokens
            bt["output_tokens"] += r.output_tokens

            bp = by_period[_bucket(r.ts, period)]
            bp["cost_usd"] += r.cost_usd
            bp["calls"] += 1

        return {
            "period": period,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "total_cost_usd": round(total_cost, 6),
            "total_calls": len(records),
            "total_tokens": {
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            "by_model": {
                k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                for k, v in sorted(by_model.items(), key=lambda x: -x[1]["cost_usd"])
            },
            "by_tag": {
                k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                for k, v in sorted(by_tag.items(), key=lambda x: -x[1]["cost_usd"])
            },
            "by_period": dict(sorted(by_period.items())),
        }

    def to_json(self, since=None, until=None, period="daily") -> str:
        return json.dumps(self.build(since, until, period), indent=2)

    def to_text(self, since=None, until=None, period="daily") -> str:
        d = self.build(since, until, period)
        lines = []
        lines.append("─" * 55)
        lines.append("  LLM Cost Report")
        if d["since"]:
            since_dt = datetime.fromisoformat(d["since"])
            lines.append(f"  Since: {since_dt.strftime('%Y-%m-%d %H:%M UTC')}")
        if d["until"]:
            until_dt = datetime.fromisoformat(d["until"])
            lines.append(f"  Until: {until_dt.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("─" * 55)

        lines.append(f"\n  Total cost  : ${d['total_cost_usd']:.4f}")
        lines.append(f"  Total calls : {d['total_calls']}")
        tok = d["total_tokens"]
        lines.append(f"  Tokens      : {tok['input']:,} in / {tok['output']:,} out ({tok['total']:,} total)")

        if d["by_model"]:
            lines.append("\n  By Model:")
            for model, stats in d["by_model"].items():
                lines.append(
                    f"    {model:<40} ${stats['cost_usd']:.4f}  ({stats['calls']} calls)"
                )

        if d["by_tag"]:
            lines.append("\n  By Tag / Project:")
            for tag, stats in d["by_tag"].items():
                lines.append(
                    f"    {tag:<30} ${stats['cost_usd']:.4f}  ({stats['calls']} calls)"
                )

        if d["by_period"]:
            lines.append(f"\n  {period.capitalize()} Breakdown:")
            for bucket, stats in d["by_period"].items():
                lines.append(
                    f"    {bucket:<15} ${stats['cost_usd']:.4f}  ({stats['calls']} calls)"
                )

        lines.append("\n" + "─" * 55)
        return "\n".join(lines)
