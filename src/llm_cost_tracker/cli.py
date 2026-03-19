#!/usr/bin/env python3
"""
llm-cost CLI — track and report LLM API spending.

Usage:
  llm-cost add --model gpt-4o --input 1000 --output 500 [--tag myproject]
  llm-cost report [--since 7d] [--format text|json] [--period daily|weekly|monthly]
  llm-cost models [--provider openai]
  llm-cost budget set --tag myproject --limit 5.00
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .pricing import ModelPricing
from .tracker import CostTracker, UsageRecord, BudgetGuard
from .report import CostReport


# ── time parsing ─────────────────────────────────────────────────────────────

def parse_since(s: str) -> datetime:
    """Parse '7d', '30d', '1w', '1m', or ISO datetime string."""
    now = datetime.now(timezone.utc)
    s = s.strip().lower()
    if s.endswith("d"):
        return now - timedelta(days=int(s[:-1]))
    if s.endswith("w"):
        return now - timedelta(weeks=int(s[:-1]))
    if s.endswith("m"):
        return now - timedelta(days=30 * int(s[:-1]))
    if s.endswith("h"):
        return now - timedelta(hours=int(s[:-1]))
    # ISO
    return datetime.fromisoformat(s)


# ── CLI commands ─────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace, tracker: CostTracker) -> None:
    record = tracker.add_call(
        model=args.model,
        input_tokens=args.input,
        output_tokens=args.output,
        tag=args.tag,
    )
    pricing = ModelPricing()
    try:
        cost = pricing.compute_cost(args.model, args.input, args.output)
        cost_str = f"${cost:.6f}"
    except KeyError:
        cost_str = "unknown (model not in pricing DB)"

    print(f"✓ Logged: {args.model}  {args.input:,} in / {args.output:,} out  cost={cost_str}  tag={args.tag}")
    print(f"  record_id={record.record_id}  db={tracker.db_path}")


def cmd_report(args: argparse.Namespace, tracker: CostTracker) -> None:
    since = parse_since(args.since) if args.since else None
    report = CostReport(tracker)

    if args.format == "json":
        print(report.to_json(since=since, period=args.period))
    else:
        print(report.to_text(since=since, period=args.period))


def cmd_models(args: argparse.Namespace) -> None:
    pricing = ModelPricing()
    models = pricing.list_models(provider=args.provider)
    if not models:
        print("No models found.")
        return
    print(f"\n{'Model':<45} {'Provider':<12} {'Input $/1M':>12} {'Output $/1M':>12}")
    print("─" * 85)
    for m in models:
        print(f"  {m.model:<43} {m.provider:<12} ${m.input_per_1m:>10.3f}  ${m.output_per_1m:>10.3f}")
    print()


def cmd_budget(args: argparse.Namespace, tracker: CostTracker) -> None:
    if args.budget_cmd == "set":
        guard = BudgetGuard(tag=args.tag, limit_usd=args.limit)
        tracker.set_budget(guard)
        print(f"✓ Budget set: tag={args.tag}  limit=${args.limit:.2f}")

        # current spend check
        spent = tracker.total_cost(tag=args.tag)
        pct = (spent / args.limit * 100) if args.limit > 0 else 0
        print(f"  Current spend: ${spent:.4f} ({pct:.1f}% of budget)")

    elif args.budget_cmd == "check":
        spent = tracker.total_cost(tag=args.tag)
        print(f"  Tag '{args.tag}': ${spent:.4f} spent")
    else:
        print("Usage: llm-cost budget set --tag TAG --limit AMOUNT")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-cost",
        description="LLM cost tracker — log API calls, report spending, guard budgets.",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to usage DB (default: ~/.config/llm-cost-tracker/usage.jsonl)"
    )

    sub = parser.add_subparsers(dest="command")

    # ── add ──────────────────────────────────────────────────────────────────
    p_add = sub.add_parser("add", help="Log an LLM API call")
    p_add.add_argument("--model", required=True, help="Model name, e.g. gpt-4o")
    p_add.add_argument("--input", type=int, required=True, help="Input token count")
    p_add.add_argument("--output", type=int, required=True, help="Output token count")
    p_add.add_argument("--tag", default="default", help="Project/tag label")
    p_add.add_argument("--timestamp", default=None, help="ISO timestamp (default: now)")

    # ── report ───────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Show cost report")
    p_rep.add_argument("--since", default="30d", help="e.g. 7d, 1w, 1m, or ISO date")
    p_rep.add_argument("--format", choices=["text", "json"], default="text")
    p_rep.add_argument("--period", choices=["daily", "weekly", "monthly"], default="daily")

    # ── models ───────────────────────────────────────────────────────────────
    p_mod = sub.add_parser("models", help="List pricing database")
    p_mod.add_argument("--provider", default=None, help="Filter by provider")

    # ── budget ───────────────────────────────────────────────────────────────
    p_bud = sub.add_parser("budget", help="Budget guard commands")
    budget_sub = p_bud.add_subparsers(dest="budget_cmd")
    p_bset = budget_sub.add_parser("set", help="Set a budget for a tag")
    p_bset.add_argument("--tag", required=True)
    p_bset.add_argument("--limit", type=float, required=True, help="Budget in USD")
    p_bchk = budget_sub.add_parser("check", help="Check spending for a tag")
    p_bchk.add_argument("--tag", required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    tracker = CostTracker(db_path=args.db)

    if args.command == "add":
        cmd_add(args, tracker)
    elif args.command == "report":
        cmd_report(args, tracker)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "budget":
        cmd_budget(args, tracker)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
