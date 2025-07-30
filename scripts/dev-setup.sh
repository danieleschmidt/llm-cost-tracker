#!/bin/bash
# Development Environment Setup Script
# Automated setup for LLM Cost Tracker development environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="LLM Cost Tracker"
PYTHON_VERSION="3.11"
POETRY_VERSION="1.7.1"

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "============================================"
    echo "  $PROJECT_NAME Development Setup"
    echo "============================================"
    echo -e "${NC}"
}

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Detected Linux OS"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "Detected macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Detected Windows (WSL/Cygwin)"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if Python is installed with correct version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python $PYTHON_VERSION"
        exit 1
    fi
    
    # Check Python version
    CURRENT_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION=$(echo "$PYTHON_VERSION" | cut -d'.' -f1,2)
    
    if [[ "$CURRENT_VERSION" == "$REQUIRED_VERSION" ]]; then
        print_success "Python $CURRENT_VERSION found"
    else
        print_warning "Python version mismatch. Found: $CURRENT_VERSION, Required: $REQUIRED_VERSION"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install Poetry if not present
install_poetry() {
    print_status "Checking Poetry installation..."
    
    if command -v poetry &> /dev/null; then
        CURRENT_POETRY_VERSION=$(poetry --version | cut -d' ' -f3)
        print_success "Poetry $CURRENT_POETRY_VERSION found"
    else
        print_status "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully"
        else
            print_error "Poetry installation failed. Please install manually."
            exit 1
        fi
    fi
}

# Configure Poetry
configure_poetry() {
    print_status "Configuring Poetry..."
    
    # Configure Poetry to create virtual environment in project directory
    poetry config virtualenvs.in-project true
    poetry config virtualenvs.prefer-active-python true
    
    print_success "Poetry configured"
}

# Install project dependencies
install_dependencies() {
    print_status "Installing project dependencies..."
    
    # Install main dependencies
    poetry install
    
    print_success "Dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    
    # Install pre-commit hooks
    poetry run pre-commit install
    poetry run pre-commit install --hook-type commit-msg
    
    # Run pre-commit on all files to ensure everything works
    print_status "Running pre-commit on all files (this may take a while)..."
    poetry run pre-commit run --all-files || {
        print_warning "Pre-commit found issues. They have been auto-fixed where possible."
        print_status "Please review the changes and commit them."
    }
    
    print_success "Pre-commit hooks configured"
}

# Setup environment files
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
            print_warning "Please edit .env file with your actual configuration values"
        else
            print_warning "No .env.example found. Creating basic .env file..."
            cat > .env << EOF
# Basic environment configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_metrics
DEBUG=true
LOG_LEVEL=INFO
EOF
            print_success "Created basic .env file"
        fi
    else
        print_success ".env file already exists"
    fi
}

# Check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        
        if command -v docker-compose &> /dev/null; then
            print_success "Docker Compose found"
        else
            print_warning "Docker Compose not found. Some development features may not work."
        fi
    else
        print_warning "Docker not found. Some development features may not work."
        print_status "Visit https://docs.docker.com/get-docker/ for installation instructions"
    fi
}

# Setup IDE configuration
setup_ide_config() {
    print_status "Setting up IDE configuration..."
    
    # VSCode configuration should already exist from the main setup
    if [ -d ".vscode" ]; then
        print_success "VSCode configuration found"
    else
        print_warning "VSCode configuration not found"
    fi
    
    # Create .editorconfig if it doesn't exist
    if [ ! -f ".editorconfig" ]; then
        cat > .editorconfig << EOF
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

[*.py]
indent_size = 4
max_line_length = 88

[*.{yml,yaml}]
indent_size = 2

[*.{md,rst}]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
EOF
        print_success "Created .editorconfig"
    else
        print_success ".editorconfig already exists"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test Poetry environment
    if poetry run python --version &> /dev/null; then
        print_success "Poetry environment working"
    else
        print_error "Poetry environment not working"
        exit 1
    fi
    
    # Test import of main module
    if poetry run python -c "import llm_cost_tracker" &> /dev/null; then
        print_success "Main module can be imported"
    else
        print_warning "Main module import failed (this may be expected if code isn't written yet)"
    fi
    
    # Test development tools
    poetry run black --version &> /dev/null && print_success "Black formatter working" || print_error "Black formatter not working"
    poetry run flake8 --version &> /dev/null && print_success "Flake8 linter working" || print_error "Flake8 linter not working"
    poetry run mypy --version &> /dev/null && print_success "MyPy type checker working" || print_error "MyPy type checker not working"
    poetry run pytest --version &> /dev/null && print_success "Pytest testing framework working" || print_error "Pytest not working"
}

# Run basic tests
run_basic_tests() {
    print_status "Running basic tests..."
    
    if [ -d "tests" ] && [ "$(ls -A tests)" ]; then
        poetry run pytest tests/ -v --tb=short || {
            print_warning "Some tests failed. This may be expected in early development."
        }
    else
        print_warning "No tests found to run"
    fi
}

# Create development shortcuts
create_shortcuts() {
    print_status "Creating development shortcuts..."
    
    # Create a simple Makefile if it doesn't exist
    if [ ! -f "Makefile" ]; then
        cat > Makefile << 'EOF'
.PHONY: help install test lint format clean run docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  run         Run the application"
	@echo "  docker-up   Start Docker services"
	@echo "  docker-down Stop Docker services"

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
	rm -rf htmlcov/ .coverage

run:
	poetry run uvicorn llm_cost_tracker.main:app --reload

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
EOF
        print_success "Created Makefile with common commands"
    else
        print_success "Makefile already exists"
    fi
}

# Print setup summary
print_summary() {
    echo -e "\n${GREEN}🎉 Development environment setup completed!${NC}\n"
    
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Edit .env file with your configuration values"
    echo "2. Start the database: make docker-up"
    echo "3. Run tests: make test"
    echo "4. Start development server: make run"
    echo "5. Open your IDE and start developing!"
    
    echo -e "\n${BLUE}Useful commands:${NC}"
    echo "  make help          - Show all available commands"
    echo "  poetry shell       - Activate virtual environment"
    echo "  poetry run pytest  - Run tests"
    echo "  poetry run black . - Format code"
    echo "  git commit         - Commit (pre-commit hooks will run)"
    
    echo -e "\n${BLUE}IDE Setup:${NC}"
    echo "  VSCode: Extensions and settings are configured"
    echo "  PyCharm: Use Poetry interpreter in .venv directory"
    
    echo -e "\n${YELLOW}Important Notes:${NC}"
    echo "  - Pre-commit hooks are enabled for code quality"
    echo "  - Virtual environment is created in .venv directory"
    echo "  - Docker services available via docker-compose"
    echo "  - Security scanning tools are configured"
    
    echo -e "\n${GREEN}Happy coding! 🚀${NC}"
}

# Main execution function
main() {
    print_header
    
    check_os
    check_python
    install_poetry
    configure_poetry
    install_dependencies
    setup_precommit
    setup_environment
    check_docker
    setup_ide_config
    create_shortcuts
    verify_installation
    run_basic_tests
    
    print_summary
}

# Handle script interruption
cleanup() {
    echo -e "\n${YELLOW}Setup interrupted${NC}"
    exit 130
}

trap cleanup INT

# Check if running from project root
if [ ! -f "pyproject.toml" ]; then
    print_error "This script must be run from the project root directory"
    exit 1
fi

# Run main function
main "$@"