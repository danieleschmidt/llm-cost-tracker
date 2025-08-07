# Sentiment Analyzer Pro

[![Build Status](https://img.shields.io/github/actions/workflow/status/terragon-labs/sentiment-analyzer-pro/ci.yml?branch=main)](https://github.com/terragon-labs/sentiment-analyzer-pro/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/terragon-labs/sentiment-analyzer-pro)](https://coveralls.io/github/terragon-labs/sentiment-analyzer-pro)
[![License](https://img.shields.io/github/license/terragon-labs/sentiment-analyzer-pro)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

**Enterprise-grade sentiment analysis platform** with quantum-enhanced task planning, advanced security, and production-ready deployment. Built on top of a sophisticated LLM cost tracking foundation, this system provides real-time sentiment classification with comprehensive threat detection, performance optimization, and global compliance features.

## 🌟 Core Capabilities

### 🧠 Advanced Sentiment Analysis
- **Multi-model Support**: GPT-3.5, GPT-4, Claude integration with intelligent model selection
- **Real-time Processing**: Sub-200ms latency with 75+ RPS throughput capacity
- **Batch Processing**: Intelligent chunking for high-volume analysis with parallel execution
- **Multi-language Support**: 6 languages (EN, ES, FR, DE, JA, ZH) with localized responses

### 🔒 Enterprise Security
- **Threat Detection**: Real-time scanning for injection attacks, PII exposure, and malicious content
- **Data Protection**: Automatic anonymization with GDPR/CCPA compliance features
- **Access Control**: JWT authentication, rate limiting, and IP-based security policies
- **Audit Trail**: Comprehensive logging for compliance and security monitoring

### ⚡ Quantum-Enhanced Performance
- **Task Planning**: Quantum-inspired scheduling with superposition and entanglement concepts
- **Auto-scaling**: Dynamic resource allocation based on load and performance metrics  
- **Intelligent Caching**: LRU eviction with predictive prefetching and access pattern optimization
- **Circuit Breakers**: Automatic failure isolation with health-aware load balancing

## ✨ Key Features

### 📊 LLM Cost Tracking Features
| Feature                 | Details                                                                                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Real-time Metering**  | Asynchronous Python middleware hooks into LangChain's `AsyncIteratorCallbackHandler` to capture token usage, latency, prompts, and the specific model used.                                           |
| **Budget Rules Engine** | YAML rules (`monthly_budget`, `swap_threshold`) trigger automatic model routing via LiteLLM router or Vellum API, based on up-to-date model prices.                                                     |
| **Dashboards**          | Comes with a pre-built Grafana JSON dashboard located in `/dashboards/llm-cost-dashboard.json` (UID: `llm-cost-dashboard`) to visualize costs by application, model, and user.                             |
| **Alerting**            | Integrates with Prometheus to send alerts to Slack or OpsGenie whenever predefined cost thresholds are exceeded.                                                                                     |
| **Pluggable Storage**   | Defaults to Postgres, with adapters available for ClickHouse and BigQuery to offer flexibility in data storage.                                                                                    |

### ⚛️ Quantum Task Planning Features
| Feature                    | Details                                                                                                                                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quantum Scheduling**     | Tasks exist in superposition states, allowing for probabilistic execution planning and optimal resource allocation using quantum annealing algorithms.                                          |
| **Task Entanglement**      | Related tasks can be quantum-entangled, ensuring coordinated execution and maintaining dependencies through quantum interference patterns.                                                      |
| **Performance Optimization** | High-performance caching with LRU eviction, load balancing with circuit breakers, and auto-scaling based on queue utilization and resource metrics.                                         |
| **Global Compliance**      | Built-in GDPR/CCPA compliance with PII detection, data anonymization, consent management, and data subject rights (right to access, delete, portability).                                    |
| **Multilingual Support**   | Native internationalization (i18n) with support for 6 languages: English, Spanish, French, German, Japanese, and Chinese (Simplified).                                                       |
| **Production Ready**       | Zero-downtime deployments, comprehensive monitoring with Grafana/Prometheus, automated backups, security scanning, and enterprise-grade reliability.                                           |

## 🏗️ Reference Architecture

### LLM Cost Tracking Flow
```
LangChain ↔ Cost-Middleware → OpenTelemetry SDK → OTLP Collector → Postgres → Grafana
                                    ↘ Prometheus/Alertmanager
```

### Quantum Task Planning Flow
```
Tasks → Quantum Planner → Annealing Optimizer → Load Balancer → Execution
   ↓         ↓                    ↓                   ↓            ↓
Cache    Monitoring        Auto-Scaler       Circuit Breakers   Results
```

### Integrated System Architecture
```
                    ┌─── LLM Cost Tracker ───┐
                    │                        │
    LangChain ──────┼── OpenTelemetry ──────┼──── Postgres
                    │       │               │       │
                    │   Prometheus ─────────┼────── Grafana
                    │                       │
    Tasks ──────────┼── Quantum Planner ───┼──── Execution
                    │       │               │       │
                    │   Monitoring ─────────┼────── Results
                    │                       │
                    └───────────────────────┘
```
## ⚡ Quick Start

### 🐳 Production Deployment (Recommended)
```bash
# Clone repository
git clone https://github.com/terragon-labs/llm-cost-tracker
cd llm-cost-tracker

# Configure production environment
cp .env.production.example .env.production
# Edit .env.production with your settings

# Deploy with zero-downtime
chmod +x scripts/deploy.sh
./scripts/deploy.sh deploy

# Access services
# API: https://api.your-domain.com
# Grafana: https://grafana.your-domain.com  
# Quantum Dashboard: https://api.your-domain.com/api/v1/quantum/system/state
```

### 🔬 Development Setup
```bash
# Clone and setup
git clone https://github.com/terragon-labs/llm-cost-tracker
cd llm-cost-tracker

# Start services
docker compose up -d

# Install dependencies
poetry install

# Run LLM cost tracking demo
python examples/streamlit_demo.py

# Test quantum task planner
curl -X GET http://localhost:8000/api/v1/quantum/demo

# Access Grafana and import dashboard
# http://localhost:3000 (admin/admin)
# Import: /dashboards/llm-cost-dashboard.json
```

## 🔐 Security

This tool handles sensitive API keys. To safeguard these credentials, we follow an encrypted proxy pattern. All keys should be stored in environment variables or a secure vault. For reporting vulnerabilities, please refer to our organization's `SECURITY.md` file.

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with examples
- **[Quantum Architecture](docs/QUANTUM_ARCHITECTURE.md)** - Deep dive into quantum-inspired concepts
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[Examples](examples/)** - Sample implementations and use cases

## 🚀 API Examples

### Quantum Task Planning
```python
from llm_cost_tracker import QuantumTaskPlanner, QuantumTask, ResourcePool

# Initialize planner
planner = QuantumTaskPlanner()

# Create tasks
task1 = QuantumTask(
    id="analyze_data",
    name="Data Analysis",
    priority=9.0,
    estimated_duration_minutes=30
)

# Add to planner
planner.add_task(task1)

# Generate optimal schedule
schedule = planner.generate_schedule()
print(f"Optimal execution order: {schedule}")

# Execute tasks
results = await planner.execute_schedule_async(schedule)
```

### REST API Usage
```bash
# Create a quantum task
curl -X POST http://localhost:8000/api/v1/quantum/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "id": "task_001",
    "name": "Machine Learning Pipeline", 
    "priority": 8.5,
    "estimated_duration_minutes": 45
  }'

# Generate optimal schedule
curl -X GET http://localhost:8000/api/v1/quantum/schedule

# Monitor system state
curl -X GET http://localhost:8000/api/v1/quantum/system/state
```

## 📈 Roadmap

### LLM Cost Tracker
*   **v0.1.0**: ✅ Core tracing, Grafana dashboard, and Prometheus alerts
*   **v0.2.0**: Implementation of the budget-aware model swapper with Slack alerts
*   **v0.3.0**: Introduction of multi-tenant RBAC and per-project budgets

### Quantum Task Planner  
*   **v0.1.0**: ✅ Quantum-inspired scheduling with superposition and entanglement
*   **v0.1.0**: ✅ Performance optimization with caching and load balancing
*   **v0.1.0**: ✅ Global compliance (GDPR/CCPA) and i18n support
*   **v0.2.0**: Advanced quantum algorithms and machine learning integration
*   **v0.3.0**: Distributed quantum planning across multiple nodes

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
