"""Tests for CostReport."""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_cost_tracker.tracker import CostTracker
from llm_cost_tracker.report import CostReport


@pytest.fixture
def populated_tracker(tmp_path):
    db = tmp_path / "usage.jsonl"
    t = CostTracker(db_path=db)
    now = datetime.now(timezone.utc)

    # spread across a few days
    for day_offset, model, tag, inp, out in [
        (0, "gpt-4o",      "chatbot", 1000, 500),
        (0, "gpt-4o-mini", "chatbot", 5000, 2000),
        (1, "gpt-4o",      "search",  2000, 1000),
        (2, "claude-3-5-sonnet-20241022", "search", 3000, 1500),
        (3, "gpt-4o",      "chatbot", 1500, 750),
    ]:
        ts = (now - timedelta(days=day_offset)).isoformat()
        from llm_cost_tracker.tracker import UsageRecord
        import json
        r = UsageRecord(model=model, input_tokens=inp, output_tokens=out, tag=tag, timestamp=ts)
        r.cost_usd = t.pricing.compute_cost(model, inp, out)
        with open(t.db_path, "a") as f:
            f.write(json.dumps(r.to_dict()) + "\n")

    return t


def test_report_build_basic(populated_tracker):
    report = CostReport(populated_tracker)
    d = report.build()
    assert d["total_calls"] == 5
    assert d["total_cost_usd"] > 0
    assert "gpt-4o" in d["by_model"]


def test_report_by_model(populated_tracker):
    report = CostReport(populated_tracker)
    d = report.build()
    assert "chatbot" in d["by_tag"]
    assert "search" in d["by_tag"]


def test_report_by_tag_totals(populated_tracker):
    report = CostReport(populated_tracker)
    d = report.build()
    chatbot_calls = d["by_tag"]["chatbot"]["calls"]
    assert chatbot_calls == 3


def test_report_daily_period(populated_tracker):
    report = CostReport(populated_tracker)
    d = report.build(period="daily")
    assert len(d["by_period"]) >= 3


def test_report_weekly_period(populated_tracker):
    report = CostReport(populated_tracker)
    d = report.build(period="weekly")
    assert len(d["by_period"]) >= 1


def test_report_json_valid(populated_tracker):
    import json
    report = CostReport(populated_tracker)
    j = report.to_json()
    parsed = json.loads(j)
    assert "total_cost_usd" in parsed


def test_report_text_contains_headers(populated_tracker):
    report = CostReport(populated_tracker)
    text = report.to_text()
    assert "LLM Cost Report" in text
    assert "By Model" in text
    assert "By Tag" in text


def test_empty_report(tmp_path):
    db = tmp_path / "empty.jsonl"
    t = CostTracker(db_path=db)
    report = CostReport(t)
    d = report.build()
    assert d["total_calls"] == 0
    assert d["total_cost_usd"] == 0.0


def test_since_filter_in_report(populated_tracker):
    now = datetime.now(timezone.utc)
    report = CostReport(populated_tracker)
    # only last 12 hours
    d = report.build(since=now - timedelta(hours=12))
    assert d["total_calls"] <= 5  # some may be excluded
