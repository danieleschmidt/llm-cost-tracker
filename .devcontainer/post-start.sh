#!/bin/bash
# Post-start script for devcontainer
# Runs every time the container starts

set -e

echo "🌟 Starting LLM Cost Tracker development session..."

# Ensure Poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || echo "⚠️ Virtual environment not found"

# Update dependencies if needed
echo "🔄 Checking for dependency updates..."
poetry install --no-interaction --quiet

# Start background services if docker-compose is available
if [ -f docker-compose.yml ]; then
    echo "🐳 Starting background services..."
    docker-compose up -d postgres redis prometheus grafana 2>/dev/null || echo "⚠️ Some services may not be available"
fi

# Display status
echo ""
echo "📊 Development Environment Status:"
echo "=================================="
echo "Python: $(python --version)"
echo "Poetry: $(poetry --version)"
echo "Git: $(git --version)"
echo "Node: $(node --version)"
echo "Docker: $(docker --version)"
echo ""

# Show useful commands
echo "🎯 Quick Commands:"
echo "=================="
echo "make help           - Show all available commands"
echo "make test           - Run test suite"
echo "make dev            - Start development server"
echo "make quality        - Run code quality checks"
echo "make docker-up      - Start all services"
echo ""

echo "✅ Ready for development!"