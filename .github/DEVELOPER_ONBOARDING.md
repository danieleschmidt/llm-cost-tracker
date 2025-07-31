# Developer Onboarding Checklist

Welcome to the LLM Cost Tracker project! This checklist will help you get set up quickly and efficiently.

## 🚀 Quick Start (5 minutes)

### Prerequisites Verification
- [ ] Python 3.11+ installed (`python --version`)
- [ ] Poetry installed (`poetry --version`)
- [ ] Docker and Docker Compose installed (`docker --version`)
- [ ] Git configured with your credentials

### Initial Setup
- [ ] Clone the repository: `git clone https://github.com/terragon-labs/llm-cost-tracker.git`
- [ ] Navigate to project: `cd llm-cost-tracker`
- [ ] Copy environment template: `cp .env.example .env`
- [ ] Start services: `docker compose up -d`
- [ ] Install dependencies: `poetry install`
- [ ] Run tests: `poetry run pytest`

## 🛠️ Development Environment (10 minutes)

### IDE Configuration
- [ ] Install recommended extensions (see `.vscode/extensions.json`)
- [ ] Configure editor settings (`.editorconfig` should be auto-detected)
- [ ] Enable format on save (Black, isort)
- [ ] Configure Python interpreter to use Poetry virtual environment

### Git Hooks Setup
- [ ] Install pre-commit hooks: `poetry run pre-commit install`
- [ ] Test hooks: `poetry run pre-commit run --all-files`
- [ ] Configure git rerere: `scripts/setup-git-rerere.sh`

### Verification Steps
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Grafana accessible: `http://localhost:3000` (admin/admin)
- [ ] Prometheus accessible: `http://localhost:9090`
- [ ] Run demo: `poetry run python examples/streamlit_demo.py`

## 📚 Knowledge Transfer (15 minutes)

### Architecture Understanding
- [ ] Read [ARCHITECTURE.md](ARCHITECTURE.md)
- [ ] Review [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [ ] Understand the data flow diagram in README.md
- [ ] Review ADRs in [docs/adr/](docs/adr/)

### Codebase Exploration
- [ ] Explore `src/llm_cost_tracker/` structure
- [ ] Review test organization in `tests/`
- [ ] Check configuration files in `config/`
- [ ] Examine monitoring setup in `dashboards/`

### Development Workflow
- [ ] Read [CONTRIBUTING.md](CONTRIBUTING.md)
- [ ] Understand branch naming conventions
- [ ] Review PR template in `.github/PULL_REQUEST_TEMPLATE/`
- [ ] Check issue templates in `.github/ISSUE_TEMPLATE/`

## 🔒 Security & Compliance (5 minutes)

### Security Setup
- [ ] Read [SECURITY.md](SECURITY.md)
- [ ] Understand credential management (never commit secrets!)
- [ ] Review security scanning in pre-commit hooks
- [ ] Check CODEOWNERS file for security-sensitive files

### Compliance Verification
- [ ] Run security audit: `scripts/security-audit.sh`
- [ ] Generate SBOM: `scripts/generate-sbom.sh`
- [ ] Verify no secrets in history: `poetry run pre-commit run trufflehog --all-files`

## 🧪 Testing & Quality (10 minutes)

### Testing Framework
- [ ] Run unit tests: `poetry run pytest tests/unit/`
- [ ] Run integration tests: `poetry run pytest tests/integration/`
- [ ] Run e2e tests: `poetry run pytest tests/e2e/`
- [ ] Check coverage: `poetry run pytest --cov=src/llm_cost_tracker`

### Code Quality
- [ ] Run linting: `poetry run flake8 src tests`
- [ ] Run type checking: `poetry run mypy src`
- [ ] Run formatting check: `poetry run black --check src tests`
- [ ] Run import sorting: `poetry run isort --check-only src tests`

### Performance Testing
- [ ] Run benchmark suite: `poetry run python scripts/performance-benchmark.py`
- [ ] Load test with Locust: `poetry run locust -f locustfile.py --headless -u 10 -r 2 -t 30s`

## 🚢 Deployment Understanding (5 minutes)

### CI/CD Pipeline
- [ ] Review workflow files in `workflow-configs-ready-to-deploy/`
- [ ] Understand deployment strategy in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- [ ] Check monitoring and alerting setup
- [ ] Review rollback procedures

### Local Development
- [ ] Test Docker build: `docker build -t llm-cost-tracker .`
- [ ] Verify docker-compose services: `docker compose ps`
- [ ] Check log aggregation: `docker compose logs -f`

## 📈 Monitoring & Observability (5 minutes)

### Dashboards
- [ ] Import Grafana dashboard from `dashboards/llm-cost-dashboard.json`
- [ ] Check Prometheus targets: `http://localhost:9090/targets`
- [ ] Review alert rules in `config/alert-rules.yml`
- [ ] Test alert webhook: `poetry run python scripts/test_alerts.py`

### Metrics & Traces
- [ ] Understand OpenTelemetry setup
- [ ] Review OTLP collector configuration
- [ ] Check trace sampling and retention policies

## ✅ Final Verification

### Integration Test
- [ ] Create a test branch: `git checkout -b test/onboarding-$(date +%s)`
- [ ] Make a small change (e.g., update a comment)
- [ ] Commit and push: `git add . && git commit -m "test: onboarding verification"`
- [ ] Verify pre-commit hooks ran successfully
- [ ] Create a test PR (then close it)

### Team Integration
- [ ] Join relevant Slack channels
- [ ] Schedule pairing session with a team member
- [ ] Add yourself to the team in README.md (optional)
- [ ] Review backlog in `backlog.yml`

## 🆘 Troubleshooting

### Common Issues
- **Port conflicts**: Check if ports 3000, 5432, 8000, 9090 are available
- **Permission errors**: Ensure Docker daemon is running and accessible
- **Poetry issues**: Try `poetry env remove python && poetry install`
- **Pre-commit failures**: Run `poetry run pre-commit run --all-files` to see details

### Getting Help
- Check [docs/SETUP_REQUIRED.md](docs/SETUP_REQUIRED.md) for known issues
- Search existing issues on GitHub
- Ask in the team Slack channel
- Tag `@terragon-labs/core-team` in PRs or issues

## 📝 Onboarding Feedback

After completing this checklist, please:
- [ ] Time how long each section took
- [ ] Note any confusing or missing steps
- [ ] Suggest improvements via PR or issue
- [ ] Update this checklist if needed

**Estimated total time: 45-60 minutes**

---

*Last updated: $(date) - Please update this checklist as the project evolves!*