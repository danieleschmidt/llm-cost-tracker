# llm-cost-tracker

Track what you actually spend on LLM APIs. No dependencies, no servers, no overhead.

## What it does

- **Log every API call** with model, token counts, and project tag
- **Calculate costs** from a built-in pricing database (OpenAI, Anthropic, Google, Cohere, Mistral, Meta — updated early 2026)
- **Daily/weekly/monthly reports** with per-model and per-project breakdowns
- **Budget guards** that warn or raise exceptions when you're burning through a limit
- **CLI** for quick logging and reporting
- **JSON output** for automation and dashboards

Stdlib only. No `pip install` required beyond Python 3.11+.

---

## Quickstart

```bash
# Clone
git clone https://github.com/danieleschmidt/llm-cost-tracker.git
cd llm-cost-tracker

# Log a call
python bin/llm-cost add --model gpt-4o --input 5000 --output 2000 --tag chatbot

# Report last 7 days
python bin/llm-cost report --since 7d
```

Or install as a package:

```bash
pip install -e .
llm-cost add --model gpt-4o --input 5000 --output 2000 --tag chatbot
llm-cost report --since 7d
```

---

## CLI Reference

### `llm-cost add`

Log an API call. Computes cost automatically from the pricing database.

```bash
llm-cost add --model gpt-4o --input 1000 --output 500
# ✓ Logged: gpt-4o  1,000 in / 500 out  cost=$0.007500  tag=default

llm-cost add --model claude-3-5-sonnet-20241022 --input 10000 --output 4000 --tag search
# ✓ Logged: claude-3-5-sonnet-20241022  10,000 in / 4,000 out  cost=$0.090000  tag=search
```

### `llm-cost report`

```bash
llm-cost report --since 7d
llm-cost report --since 30d --period weekly
llm-cost report --since 1d --format json
```

Example output:

```
───────────────────────────────────────────────────────
  LLM Cost Report
  Since: 2026-03-12 00:00 UTC
───────────────────────────────────────────────────────

  Total cost  : $0.1401
  Total calls : 4
  Tokens      : 85,000 in / 34,000 out (119,000 total)

  By Model:
    claude-3-5-sonnet-20241022               $0.0900  (1 calls)
    gpt-4o                                   $0.0325  (1 calls)
    gemini-1.5-flash                         $0.0097  (1 calls)
    gpt-4o-mini                              $0.0078  (1 calls)

  By Tag / Project:
    search                         $0.0900  (1 calls)
    chatbot                        $0.0403  (2 calls)
    experiments                    $0.0097  (1 calls)

  Daily Breakdown:
    2026-03-19      $0.1401  (4 calls)
───────────────────────────────────────────────────────
```

### `llm-cost models`

```bash
llm-cost models
llm-cost models --provider anthropic
```

### `llm-cost budget`

```bash
# Set a $5 limit for the "chatbot" project tag
llm-cost budget set --tag chatbot --limit 5.00

# Check current spending
llm-cost budget check --tag chatbot
```

---

## Python API

```python
from llm_cost_tracker import CostTracker, BudgetGuard, CostReport

tracker = CostTracker()

# Log a call
tracker.add_call("gpt-4o", input_tokens=1000, output_tokens=500, tag="my-project")

# Set a budget (raises BudgetExceeded when limit hit)
tracker.set_budget(BudgetGuard(tag="my-project", limit_usd=10.00, alert_at=0.80))

# Report
report = CostReport(tracker)
print(report.to_text(period="daily"))

# JSON for automation
import json
data = json.loads(report.to_json())
print(f"Total: ${data['total_cost_usd']:.4f}")
```

### Programmatic usage with `UsageRecord`

```python
from llm_cost_tracker import UsageRecord, CostTracker

tracker = CostTracker()

# Full control over record fields
record = UsageRecord(
    model="gpt-4o",
    input_tokens=2500,
    output_tokens=800,
    tag="summarizer",
)
tracker.add(record)
```

---

## Pricing Database

Built-in rates (USD per 1M tokens, early 2026):

| Model | Input | Output |
|---|---|---|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| o3-mini | $1.10 | $4.40 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-5-haiku | $0.80 | $4.00 |
| claude-3-opus | $15.00 | $75.00 |
| gemini-1.5-flash | $0.075 | $0.30 |
| gemini-2.0-flash | $0.10 | $0.40 |
| command-r-plus | $2.50 | $10.00 |
| mistral-large-2411 | $2.00 | $6.00 |

Add custom rates:
```python
from llm_cost_tracker import ModelPricing, ModelRate
pricing = ModelPricing(custom_rates={
    "my-fine-tune": ModelRate("openai", "my-fine-tune", 5.00, 15.00)
})
```

---

## Storage

Records are stored as newline-delimited JSON at:
```
~/.config/llm-cost-tracker/usage.jsonl
```

Override with the `LLM_COST_DB` env var or `--db` flag:
```bash
LLM_COST_DB=/path/to/myproject.jsonl llm-cost report --since 7d
```

Each line looks like:
```json
{"model": "gpt-4o", "input_tokens": 5000, "output_tokens": 2000, "timestamp": "2026-03-19T12:00:00+00:00", "tag": "chatbot", "record_id": "34788e63", "cost_usd": 0.0325}
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

34 tests, 0 dependencies beyond stdlib + pytest.

---

## License

MIT
