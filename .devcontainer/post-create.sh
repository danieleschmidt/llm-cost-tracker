#!/bin/bash
# Post-create script for devcontainer
# Sets up the development environment after container creation

set -e

echo "🚀 Setting up LLM Cost Tracker development environment..."

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Configure Poetry
echo "🔧 Configuring Poetry..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install dependencies
echo "📚 Installing Python dependencies..."
poetry install

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
poetry run pre-commit install

# Setup git configuration
echo "🐙 Configuring Git..."
git config --global --add safe.directory /workspace

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p reports

# Install additional tools
echo "🛠️ Installing additional tools..."
npm install -g @semantic-release/changelog @semantic-release/git @semantic-release/exec

# Setup environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env 2>/dev/null || echo "⚠️ No .env.example found"
fi

# Initialize database (if docker-compose is running)
echo "🗄️ Checking database setup..."
if docker-compose ps postgres | grep -q "Up"; then
    echo "Database is running, applying migrations..."
    poetry run alembic upgrade head 2>/dev/null || echo "⚠️ Migrations not yet available"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Start services: make docker-up"
echo "  2. Run tests: make test"
echo "  3. Start development server: make dev"
echo "  4. Open browser: http://localhost:8000"
echo ""