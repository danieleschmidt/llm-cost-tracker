#!/bin/bash
set -e

echo "🚀 Setting up LLM Cost Tracker development environment..."

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"

# Install project dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Setup git configuration for container
git config --global --add safe.directory /workspaces/llm-cost-tracker
git config --global init.defaultBranch main
git config --global pull.rebase false

# Install additional development tools
pip install --user poetry-plugin-export

# Create necessary directories
mkdir -p docs/{adr,guides,runbooks,status}
mkdir -p .github/{workflows,ISSUE_TEMPLATE,PULL_REQUEST_TEMPLATE}
mkdir -p tests/{unit,integration,e2e}

# Install Docker Compose if not available
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Generate environment file from template
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || echo "Creating .env file..."
    cat > .env << 'EOF'
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=llm_costs
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# OpenTelemetry Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=llm-cost-tracker

# LLM Provider API Keys (add your keys here)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=admin
PROMETHEUS_RETENTION=30d

# Security
SECRET_KEY=dev_secret_key_change_in_production
DEBUG=true
EOF
fi

echo "✅ Development environment setup complete!"
echo "🔧 Next steps:"
echo "   1. Add your API keys to .env file"
echo "   2. Run: docker-compose up -d"
echo "   3. Run: poetry run python examples/streamlit_demo.py"