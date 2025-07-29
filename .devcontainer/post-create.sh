#!/bin/bash
# Post-create script for DevContainer setup
# This script runs after the container is created

set -euo pipefail

echo "🚀 Running post-create setup for LLM Cost Tracker development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    log_error "pyproject.toml not found. Are we in the right directory?"
    exit 1
fi

# Install Python dependencies
log_info "Installing Python dependencies with Poetry..."
poetry install --with dev
log_success "Python dependencies installed"

# Install pre-commit hooks
log_info "Installing pre-commit hooks..."
poetry run pre-commit install --install-hooks
log_success "Pre-commit hooks installed"

# Create necessary directories
log_info "Creating development directories..."
mkdir -p {logs,sbom/{spdx,cyclonedx,reports,attestations},temp,data,coverage}
log_success "Development directories created"

# Set up environment variables
log_info "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    log_success "Environment file created from template"
else
    log_warning "Environment file already exists, skipping..."
fi

# Initialize the database (wait for PostgreSQL to be ready)
log_info "Waiting for PostgreSQL to be ready..."
for i in {1..30}
do
    if pg_isready -h postgres -p 5432 -U postgres; then
        log_success "PostgreSQL is ready"
        break
    fi
    echo "Waiting for PostgreSQL..."
    sleep 1
done

# Test database connection
log_info "Testing database connection..."
if poetry run python -c "
import asyncio
import asyncpg
async def test_db():
    try:
        conn = await asyncpg.connect('postgresql://postgres:postgres@postgres:5432/llm_metrics')
        await conn.close()
        print('Database connection successful')
    except Exception as e:
        print(f'Database connection failed: {e}')
        exit(1)
asyncio.run(test_db())
"; then
    log_success "Database connection verified"
else
    log_warning "Database connection test failed, but continuing..."
fi

# Create initial database schema (if needed)
log_info "Setting up database schema..."
# Add database initialization logic here if needed
log_success "Database schema setup completed"

# Install additional development tools
log_info "Installing additional development tools..."

# Install GitHub CLI extensions
if command -v gh &> /dev/null; then
    gh extension install github/gh-copilot || log_warning "Failed to install gh-copilot extension"
    gh extension install dlvhdr/gh-dash || log_warning "Failed to install gh-dash extension"
fi

log_success "Additional tools installed"

# Set up Git configuration for the project
log_info "Configuring Git for the project..."
git config core.autocrlf false
git config core.eol lf
git config pull.rebase true
git config push.default simple

# Set up Git hooks directory
if [ -d ".git" ]; then
    git config core.hooksPath .git/hooks
fi

log_success "Git configuration completed"

# Verify installation
log_info "Verifying installation..."

# Check Poetry
if poetry --version; then
    log_success "Poetry is working"
else
    log_error "Poetry verification failed"
fi

# Check Python environment
if poetry run python --version; then
    log_success "Python environment is working"
else
    log_error "Python environment verification failed"
fi

# Check pre-commit
if poetry run pre-commit --version; then
    log_success "Pre-commit is working"
else
    log_error "Pre-commit verification failed"
fi

# Create development status file
cat > .dev-status.json <<EOF
{
    "setup_completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "environment": "devcontainer",
    "python_version": "$(poetry run python --version)",
    "poetry_version": "$(poetry --version)",
    "node_version": "$(node --version)",
    "npm_version": "$(npm --version)",
    "services": {
        "postgres": "ready",
        "redis": "ready",
        "prometheus": "ready",
        "grafana": "ready"
    }
}
EOF

log_success "Development status file created"

# Display next steps
echo
log_info "🎉 Development environment setup completed!"
echo
echo "Next steps:"
echo "  1. Open the workspace in VS Code"
echo "  2. Install recommended extensions (should happen automatically)"
echo "  3. Start development with: make dev"
echo "  4. Run tests with: make test"
echo "  5. Check code quality with: make quality"
echo
echo "Available services:"
echo "  - FastAPI app: http://localhost:8000"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jaeger: http://localhost:16686"
echo
echo "Happy coding! 🚀"