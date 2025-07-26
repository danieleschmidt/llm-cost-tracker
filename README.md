# llm-cost-tracker

[![Build Status](https://img.shields.io/github/actions/workflow/status/terragon-labs/llm-cost-tracker/ci.yml?branch=main)](https://github.com/terragon-labs/llm-cost-tracker/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/terragon-labs/llm-cost-tracker)](https://coveralls.io/github/terragon-labs/llm-cost-tracker)
[![License](https://img.shields.io/github/license/terragon-labs/llm-cost-tracker)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

A self-hostable OpenTelemetry collector and rules engine designed to capture token, latency, and cost data from LangChain and LiteLLM callbacks. This tool streams the collected data to Postgres and visualizes it using Grafana. It combines proxying functionalities similar to Helicone with a budget-guard feature that can automatically switch to more cost-effective models by leveraging Vellum's price catalog.

## ✨ Key Features

| Feature                 | Details                                                                                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Real-time Metering**  | Asynchronous Python middleware hooks into LangChain's `AsyncIteratorCallbackHandler` to capture token usage, latency, prompts, and the specific model used.                                           |
| **Budget Rules Engine** | YAML rules (`monthly_budget`, `swap_threshold`) trigger automatic model routing via LiteLLM router or Vellum API, based on up-to-date model prices.                                                     |
| **Dashboards**          | Comes with a pre-built Grafana JSON dashboard located in `/dashboards/llm-cost-dashboard.json` (UID: `llm-cost-dashboard`) to visualize costs by application, model, and user.                             |
| **Alerting**            | Integrates with Prometheus to send alerts to Slack or OpsGenie whenever predefined cost thresholds are exceeded.                                                                                     |
| **Pluggable Storage**   | Defaults to Postgres, with adapters available for ClickHouse and BigQuery to offer flexibility in data storage.                                                                                    |

## 🏗️ Reference Architecture
Use code with caution.
Markdown
LangChain ↔ Cost-Middleware → OpenTelemetry SDK → OTLP Collector → Postgres → Grafana
↘ Prometheus/Alertmanager
Generated code
## ⚡ Quick Start

1.  Clone the repository: `git clone https://github.com/terragon-labs/llm-cost-tracker`
2.  Navigate to the project directory: `cd llm-cost-tracker`
3.  Start the services: `docker compose up -d`
4.  Install Python packages: `poetry install`
5.  Run the demo to send traces: `python examples/streamlit_demo.py`
6.  Open Grafana (`http://localhost:3000`) and import the dashboard from `/dashboards/llm-cost-dashboard.json`.

## 🔐 Security

This tool handles sensitive API keys. To safeguard these credentials, we follow an encrypted proxy pattern. All keys should be stored in environment variables or a secure vault. For reporting vulnerabilities, please refer to our organization's `SECURITY.md` file.

## 📈 Roadmap

*   **v0.1.0**: Core tracing, Grafana dashboard, and Prometheus alerts.
*   **v0.2.0**: Implementation of the budget-aware model swapper with Slack alerts.
*   **v0.3.0**: Introduction of multi-tenant RBAC and per-project budgets.

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` for guidelines and our `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained for version history.

## See Also

*   **[lang-observatory](../lang-observatory)**: Integrates these cost metrics into a unified observability stack.
*   **[eval-genius-agent-bench](../eval-genius-agent-bench)**: Uses this tracker to overlay cost data on performance evaluations.

## 📝 Licenses & Attribution

This project is licensed under the Apache-2.0 License. It incorporates functionalities inspired by Helicone, which is licensed under the MIT License. A copy of relevant downstream licenses can be found in the `LICENSES/` directory.

## 📚 References

*   **LangChain Callbacks**: [AsyncIteratorCallbackHandler Docs](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.streaming_iterator.AsyncIteratorCallbackHandler.html)
*   **Vellum LLM Cost Comparison**: [Vellum AI Blog](https://www.vellum.ai/blog/llm-cost-comparison)
*   **Helicone**: [Official Site](https://www.helicone.ai/)
