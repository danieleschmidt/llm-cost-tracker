# LLM Cost Tracker - Development Makefile
.PHONY: help install dev test lint format typecheck security clean build docker-build docker-up docker-down docs

# Variables
PYTHON := poetry run python
PYTEST := poetry run pytest
POETRY := poetry

help: ## Show this help message
	@echo "LLM Cost Tracker - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make install && make docker-up && make dev"

# Development Environment
install: ## Install project dependencies
	$(POETRY) install
	$(POETRY) run pre-commit install

update: ## Update project dependencies
	$(POETRY) update
	$(POETRY) run pre-commit autoupdate

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Code Quality
format: ## Format code with black and isort
	$(POETRY) run black src tests
	$(POETRY) run isort src tests

lint: ## Run linting checks
	$(POETRY) run flake8 src tests
	$(POETRY) run black --check src tests
	$(POETRY) run isort --check-only src tests

typecheck: ## Run type checking with mypy
	$(POETRY) run mypy src

security: ## Run security checks
	$(POETRY) run bandit -r src/
	$(POETRY) run safety check

quality: lint typecheck security ## Run all quality checks

# Testing
test: ## Run test suite
	$(PYTEST) tests/ --cov=src/llm_cost_tracker --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ --cov=src/llm_cost_tracker -f

test-coverage: ## Generate coverage report
	$(PYTEST) tests/ --cov=src/llm_cost_tracker --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Development
dev: ## Start development server
	$(PYTHON) -m uvicorn llm_cost_tracker.main:app --reload --host 0.0.0.0 --port 8000

demo: ## Run Streamlit demo
	$(PYTHON) examples/streamlit_demo.py

shell: ## Start Python shell with project context
	$(POETRY) shell

# Build and Packaging
build: ## Build the package
	$(POETRY) build

publish: ## Publish to PyPI (requires authentication)
	$(POETRY) publish

# Docker Operations
docker-build: ## Build Docker image
	docker build -t llm-cost-tracker:latest .

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker resources
	docker-compose down -v
	docker system prune -f

# Database Operations
db-migrate: ## Run database migrations
	$(PYTHON) -m alembic upgrade head

db-reset: ## Reset database
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS llm_metrics;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE llm_metrics;"
	$(PYTHON) -m alembic upgrade head

# Documentation
docs: ## Generate documentation
	$(POETRY) run sphinx-build -b html docs/ docs/_build/

docs-serve: ## Serve documentation locally
	$(POETRY) run python -m http.server 8080 --directory docs/_build/

# Monitoring and Health
health-check: ## Check service health
	curl -f http://localhost:8000/health || exit 1

metrics: ## View current metrics
	curl -s http://localhost:8000/metrics

alerts: ## Test alerting system
	$(PYTHON) scripts/test_alerts.py

# Autonomous System
autonomous: ## Run autonomous development system
	$(PYTHON) autonomous_senior_assistant.py run --max-iterations 10

autonomous-discover: ## Run discovery phase only
	$(PYTHON) autonomous_senior_assistant.py discover

backlog: ## Show current backlog status
	$(PYTHON) -c "from src.llm_cost_tracker.backlog_manager import BacklogManager; BacklogManager().show_status()"

# CI/CD Simulation
ci: ## Simulate CI pipeline locally
	make quality
	make test
	make build
	make security
	@echo "✅ CI pipeline completed successfully"

# Release Preparation
pre-release: ## Prepare for release
	make clean
	make ci
	make docker-build
	@echo "✅ Ready for release"

# Environment Setup
setup-dev: ## Setup development environment
	@echo "🚀 Setting up development environment..."
	make install
	make docker-up
	@echo "✅ Development environment ready!"
	@echo "📝 Next steps:"
	@echo "   1. Copy .env.example to .env and configure"
	@echo "   2. Run: make dev"
	@echo "   3. Visit: http://localhost:8000"

# Troubleshooting
logs: ## View application logs
	tail -f logs/app.log

debug: ## Start in debug mode
	$(PYTHON) -m debugpy --listen 5678 --wait-for-client -m uvicorn llm_cost_tracker.main:app --reload

# Performance
benchmark: ## Run performance benchmarks
	$(PYTHON) -m pytest tests/performance/ --benchmark-only

profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats -m uvicorn llm_cost_tracker.main:app