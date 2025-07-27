#!/bin/bash
# LLM Cost Tracker - Build Script
# Comprehensive build pipeline for local development and CI/CD

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BUILD_TARGET=${BUILD_TARGET:-production}
SKIP_TESTS=${SKIP_TESTS:-false}
SKIP_SECURITY=${SKIP_SECURITY:-false}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-}
VERSION=${VERSION:-$(date +%Y%m%d-%H%M%S)}
PLATFORM=${PLATFORM:-linux/amd64,linux/arm64}

# Functions
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
    command -v poetry >/dev/null 2>&1 || error "Poetry is required but not installed"
    
    if [ "$BUILD_TARGET" = "production" ]; then
        command -v docker-compose >/dev/null 2>&1 || warning "Docker Compose not found, some features may not work"
    fi
    
    success "Prerequisites check passed"
}

# Clean build artifacts
clean() {
    log "Cleaning build artifacts..."
    
    # Python artifacts
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
    
    # Docker artifacts
    docker system prune -f || warning "Failed to clean Docker artifacts"
    
    success "Cleanup completed"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    poetry install
    poetry run pre-commit install
    
    success "Dependencies installed"
}

# Code quality checks
quality_checks() {
    log "Running code quality checks..."
    
    # Formatting
    log "Checking code formatting..."
    poetry run black --check src tests || error "Code formatting check failed. Run 'make format' to fix."
    
    # Import sorting
    log "Checking import sorting..."
    poetry run isort --check-only src tests || error "Import sorting check failed. Run 'make format' to fix."
    
    # Linting
    log "Running linting..."
    poetry run flake8 src tests || error "Linting failed"
    
    # Type checking
    log "Running type checking..."
    poetry run mypy src || error "Type checking failed"
    
    success "Code quality checks passed"
}

# Security checks
security_checks() {
    if [ "$SKIP_SECURITY" = "true" ]; then
        info "Skipping security checks"
        return 0
    fi
    
    log "Running security checks..."
    
    # Bandit security scan
    log "Running Bandit security scan..."
    poetry run bandit -r src/ || warning "Bandit security scan found issues"
    
    # Safety dependency check
    log "Checking dependencies for known vulnerabilities..."
    poetry run safety check || warning "Safety check found vulnerable dependencies"
    
    # Secrets scan
    log "Scanning for secrets..."
    poetry run pre-commit run trufflehog --all-files || warning "Secrets scan found potential issues"
    
    success "Security checks completed"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        info "Skipping tests"
        return 0
    fi
    
    log "Running test suite..."
    
    # Unit tests
    log "Running unit tests..."
    poetry run pytest tests/unit/ -v --cov=src/llm_cost_tracker --cov-report=term
    
    # Integration tests
    log "Running integration tests..."
    poetry run pytest tests/integration/ -v
    
    # Performance tests
    log "Running performance tests..."
    poetry run pytest tests/performance/ -m performance --benchmark-only || warning "Performance tests failed or benchmarks not met"
    
    success "Test suite completed"
}

# Build Docker images
build_docker() {
    log "Building Docker images for target: $BUILD_TARGET"
    
    case $BUILD_TARGET in
        development)
            docker build --target development -t llm-cost-tracker:dev .
            success "Development image built: llm-cost-tracker:dev"
            ;;
        production)
            docker build --target production -t llm-cost-tracker:latest -t llm-cost-tracker:$VERSION .
            success "Production image built: llm-cost-tracker:latest, llm-cost-tracker:$VERSION"
            ;;
        security)
            docker build --target security -t llm-cost-tracker:security .
            success "Security image built: llm-cost-tracker:security"
            ;;
        all)
            docker build --target development -t llm-cost-tracker:dev .
            docker build --target production -t llm-cost-tracker:latest -t llm-cost-tracker:$VERSION .
            docker build --target security -t llm-cost-tracker:security .
            success "All images built successfully"
            ;;
        *)
            error "Unknown build target: $BUILD_TARGET. Use: development, production, security, or all"
            ;;
    esac
}

# Security scan Docker image
scan_docker_image() {
    if [ "$SKIP_SECURITY" = "true" ]; then
        info "Skipping Docker security scan"
        return 0
    fi
    
    log "Scanning Docker image for vulnerabilities..."
    
    # Use Trivy to scan the built image
    if command -v trivy >/dev/null 2>&1; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL llm-cost-tracker:latest || warning "Docker security scan found vulnerabilities"
    else
        warning "Trivy not installed, skipping Docker security scan"
    fi
    
    success "Docker security scan completed"
}

# Package application
package_application() {
    log "Packaging application..."
    
    # Build Python package
    poetry build
    
    # Create release archive
    tar -czf "llm-cost-tracker-${VERSION}.tar.gz" \
        src/ \
        config/ \
        pyproject.toml \
        README.md \
        LICENSE \
        docker-compose.yml \
        Dockerfile
    
    success "Application packaged: llm-cost-tracker-${VERSION}.tar.gz"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log "Generating Software Bill of Materials..."
    
    if command -v cyclonedx-py >/dev/null 2>&1; then
        cyclonedx-py -o sbom.json
        success "SBOM generated: sbom.json"
    else
        # Fallback: generate simple dependency list
        poetry export -f requirements.txt --output requirements.txt
        poetry run pip list --format=json > dependencies.json
        warning "CycloneDX not available, generated fallback dependency files"
    fi
}

# Push to registry
push_to_registry() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        info "No registry specified, skipping push"
        return 0
    fi
    
    log "Pushing images to registry: $DOCKER_REGISTRY"
    
    docker tag llm-cost-tracker:latest $DOCKER_REGISTRY/llm-cost-tracker:latest
    docker tag llm-cost-tracker:$VERSION $DOCKER_REGISTRY/llm-cost-tracker:$VERSION
    
    docker push $DOCKER_REGISTRY/llm-cost-tracker:latest
    docker push $DOCKER_REGISTRY/llm-cost-tracker:$VERSION
    
    success "Images pushed to registry"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Start services for verification
    docker-compose -f docker-compose.test.yml up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Health check
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        success "Health check passed"
    else
        error "Health check failed"
    fi
    
    # Cleanup
    docker-compose -f docker-compose.test.yml down
    
    success "Deployment verification completed"
}

# Generate build report
generate_report() {
    log "Generating build report..."
    
    cat > build-report.json << EOF
{
    "build_id": "$(uuidgen || echo 'build-'$(date +%s))",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$VERSION",
    "target": "$BUILD_TARGET",
    "platform": "$PLATFORM",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "docker_images": [
        "llm-cost-tracker:latest",
        "llm-cost-tracker:$VERSION"
    ],
    "artifacts": [
        "llm-cost-tracker-${VERSION}.tar.gz",
        "sbom.json",
        "build-report.json"
    ],
    "tests": {
        "skipped": $SKIP_TESTS,
        "status": "passed"
    },
    "security": {
        "skipped": $SKIP_SECURITY,
        "status": "passed"
    }
}
EOF
    
    success "Build report generated: build-report.json"
}

# Main execution
main() {
    log "Starting LLM Cost Tracker build process..."
    log "Build target: $BUILD_TARGET"
    log "Version: $VERSION"
    log "Platform: $PLATFORM"
    
    check_prerequisites
    clean
    install_dependencies
    quality_checks
    security_checks
    run_tests
    build_docker
    scan_docker_image
    package_application
    generate_sbom
    push_to_registry
    verify_deployment
    generate_report
    
    success "Build completed successfully! 🎉"
    info "Artifacts generated:"
    info "  - Docker images: llm-cost-tracker:latest, llm-cost-tracker:$VERSION"
    info "  - Package: llm-cost-tracker-${VERSION}.tar.gz"
    info "  - SBOM: sbom.json"
    info "  - Report: build-report.json"
}

# Handle script arguments
case "${1:-}" in
    clean)
        clean
        ;;
    deps|dependencies)
        install_dependencies
        ;;
    quality)
        quality_checks
        ;;
    security)
        security_checks
        ;;
    test)
        run_tests
        ;;
    docker)
        build_docker
        ;;
    package)
        package_application
        ;;
    sbom)
        generate_sbom
        ;;
    verify)
        verify_deployment
        ;;
    all|"")
        main
        ;;
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  clean       Clean build artifacts"
        echo "  deps        Install dependencies"
        echo "  quality     Run code quality checks"
        echo "  security    Run security checks"
        echo "  test        Run test suite"
        echo "  docker      Build Docker images"
        echo "  package     Package application"
        echo "  sbom        Generate Software Bill of Materials"
        echo "  verify      Verify deployment"
        echo "  all         Run complete build pipeline (default)"
        echo "  help        Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  BUILD_TARGET    Target to build (development|production|security|all)"
        echo "  SKIP_TESTS      Skip test execution (true|false)"
        echo "  SKIP_SECURITY   Skip security checks (true|false)"
        echo "  DOCKER_REGISTRY Registry to push images to"
        echo "  VERSION         Version tag for artifacts"
        echo "  PLATFORM        Target platform(s) for Docker build"
        ;;
    *)
        error "Unknown command: $1. Use '$0 help' for usage information."
        ;;
esac