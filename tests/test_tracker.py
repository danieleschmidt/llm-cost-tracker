"""Tests for CostTracker and BudgetGuard."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_cost_tracker.tracker import (
    CostTracker, UsageRecord, BudgetGuard, BudgetExceeded
)


@pytest.fixture
def tmp_tracker(tmp_path):
    db = tmp_path / "usage.jsonl"
    return CostTracker(db_path=db)


def test_add_and_retrieve(tmp_tracker):
    r = tmp_tracker.add_call("gpt-4o", 1000, 500, tag="proj-a")
    assert r.cost_usd > 0
    records = tmp_tracker.records()
    assert len(records) == 1
    assert records[0].model == "gpt-4o"


def test_cost_calculation(tmp_tracker):
    # gpt-4o: $2.50/1M in + $10.00/1M out
    # 100k in + 50k out = $0.25 + $0.50 = $0.75
    r = tmp_tracker.add_call("gpt-4o", 100_000, 50_000)
    assert abs(r.cost_usd - 0.75) < 1e-4


def test_multiple_records(tmp_tracker):
    tmp_tracker.add_call("gpt-4o", 1000, 1000, tag="a")
    tmp_tracker.add_call("gpt-4o-mini", 1000, 1000, tag="b")
    tmp_tracker.add_call("gpt-4o", 500, 500, tag="a")
    assert len(tmp_tracker.records()) == 3
    assert len(tmp_tracker.records(tag="a")) == 2


def test_tag_filter(tmp_tracker):
    tmp_tracker.add_call("gpt-4o", 1000, 1000, tag="alpha")
    tmp_tracker.add_call("gpt-4o", 1000, 1000, tag="beta")
    alpha = tmp_tracker.records(tag="alpha")
    assert len(alpha) == 1
    assert alpha[0].tag == "alpha"


def test_since_filter(tmp_tracker):
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=10)).isoformat()
    tmp_tracker.add_call("gpt-4o", 1000, 500, tag="old")
    # manually write old record
    r = UsageRecord("gpt-4o", 1000, 500, timestamp=old, tag="very-old")
    with open(tmp_tracker.db_path, "a") as f:
        f.write(json.dumps(r.to_dict()) + "\n")

    since_5d = now - timedelta(days=5)
    recent = tmp_tracker.records(since=since_5d)
    assert all(rec.tag != "very-old" for rec in recent)


def test_total_cost(tmp_tracker):
    tmp_tracker.add_call("gpt-4o-mini", 10_000, 10_000)
    tmp_tracker.add_call("gpt-4o-mini", 10_000, 10_000)
    total = tmp_tracker.total_cost()
    # gpt-4o-mini: $0.15/1M in + $0.60/1M out
    expected = 2 * (0.15 * 10_000 / 1_000_000 + 0.60 * 10_000 / 1_000_000)
    assert abs(total - expected) < 1e-8


def test_total_tokens(tmp_tracker):
    tmp_tracker.add_call("gpt-4o", 100, 200)
    tmp_tracker.add_call("gpt-4o", 300, 400)
    tok = tmp_tracker.total_tokens()
    assert tok["input"] == 400
    assert tok["output"] == 600
    assert tok["total"] == 1000


def test_unknown_model_logged_zero_cost(tmp_tracker):
    r = tmp_tracker.add_call("unknown-model-xyz", 1000, 1000)
    assert r.cost_usd == 0.0
    records = tmp_tracker.records()
    assert len(records) == 1


def test_usage_record_serialization():
    r = UsageRecord("gpt-4o", 1000, 500, tag="test")
    d = r.to_dict()
    r2 = UsageRecord.from_dict(d)
    assert r2.model == r.model
    assert r2.input_tokens == r.input_tokens
    assert r2.tag == r.tag


def test_budget_guard_ok():
    guard = BudgetGuard(tag="proj", limit_usd=10.0)
    result = guard.check(5.0)
    assert result is None


def test_budget_guard_warn():
    guard = BudgetGuard(tag="proj", limit_usd=10.0, alert_at=0.8)
    result = guard.check(8.5)
    assert result == "warn"


def test_budget_guard_exceeded_raises():
    guard = BudgetGuard(tag="proj", limit_usd=10.0, raise_on_exceed=True)
    with pytest.raises(BudgetExceeded) as exc_info:
        guard.check(11.0)
    assert exc_info.value.tag == "proj"
    assert exc_info.value.spent == 11.0


def test_budget_guard_exceeded_no_raise():
    guard = BudgetGuard(tag="proj", limit_usd=10.0, raise_on_exceed=False)
    result = guard.check(11.0)
    assert result == "exceeded"


def test_tracker_budget_integration(tmp_tracker):
    guard = BudgetGuard(tag="expensive", limit_usd=0.000001, raise_on_exceed=True)
    tmp_tracker.set_budget(guard)
    with pytest.raises(BudgetExceeded):
        tmp_tracker.add_call("gpt-4o", 10_000, 10_000, tag="expensive")


def test_empty_db_returns_nothing(tmp_tracker):
    assert tmp_tracker.records() == []
    assert tmp_tracker.total_cost() == 0.0
