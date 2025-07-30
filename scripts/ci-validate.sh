#!/bin/bash
# CI Validation Script
# Runs the same checks as CI/CD pipeline locally before committing

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="ci-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED_CHECKS=()

# Create results directory
mkdir -p "$RESULTS_DIR"

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
    echo "     CI/CD Pipeline Validation"
    echo "============================================"
    echo -e "${NC}"
}

# Track failed checks
add_failed_check() {
    FAILED_CHECKS+=("$1")
}

# Check if poetry is available
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed. Please install Poetry first."
        exit 1
    fi
    
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Run from project root."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    poetry install --with dev
    print_success "Dependencies installed"
}

# Code formatting check
check_formatting() {
    print_status "Checking code formatting with Black..."
    
    if poetry run black --check src tests > "$RESULTS_DIR/black_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Code formatting check passed"
        return 0
    else
        print_error "Code formatting check failed"
        print_warning "Run 'poetry run black src tests' to fix formatting issues"
        add_failed_check "formatting"
        return 1
    fi
}

# Import sorting check
check_import_sorting() {
    print_status "Checking import sorting with isort..."
    
    if poetry run isort --check-only src tests > "$RESULTS_DIR/isort_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Import sorting check passed"
        return 0
    else
        print_error "Import sorting check failed"
        print_warning "Run 'poetry run isort src tests' to fix import sorting"
        add_failed_check "import-sorting"
        return 1
    fi
}

# Linting check
check_linting() {
    print_status "Running linting with Flake8..."
    
    if poetry run flake8 src tests > "$RESULTS_DIR/flake8_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Linting check passed"
        return 0
    else
        print_error "Linting check failed"
        print_warning "Check $RESULTS_DIR/flake8_check_$TIMESTAMP.txt for details"
        add_failed_check "linting"
        return 1
    fi
}

# Type checking
check_type_checking() {
    print_status "Running type checking with MyPy..."
    
    if poetry run mypy src > "$RESULTS_DIR/mypy_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Type checking passed"
        return 0
    else
        print_error "Type checking failed"
        print_warning "Check $RESULTS_DIR/mypy_check_$TIMESTAMP.txt for details"
        add_failed_check "type-checking"
        return 1
    fi
}

# Security checks
check_security() {
    print_status "Running security checks..."
    
    local security_passed=true
    
    # Bandit security scan
    if poetry run bandit -r src/ > "$RESULTS_DIR/bandit_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Bandit security scan passed"
    else
        print_error "Bandit security scan failed"
        security_passed=false
    fi
    
    # Safety dependency check
    if poetry run safety check > "$RESULTS_DIR/safety_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Safety dependency check passed"
    else
        print_error "Safety dependency check failed"
        security_passed=false
    fi
    
    if [ "$security_passed" = true ]; then
        return 0
    else
        add_failed_check "security"
        return 1
    fi
}

# Run tests
run_tests() {
    print_status "Running test suite..."
    
    # Check if tests directory exists and has content
    if [ ! -d "tests" ] || [ -z "$(ls -A tests)" ]; then
        print_warning "No tests found to run"
        return 0
    fi
    
    if poetry run pytest tests/ -v --cov=src/llm_cost_tracker --cov-report=html --cov-report=term > "$RESULTS_DIR/pytest_check_$TIMESTAMP.txt" 2>&1; then
        print_success "All tests passed"
        
        # Check coverage
        if command -v coverage &> /dev/null; then
            COVERAGE=$(poetry run coverage report --show-missing | grep TOTAL | awk '{print $4}' | sed 's/%//')
            if [ -n "$COVERAGE" ]; then
                if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
                    print_success "Test coverage: $COVERAGE% (Good)"
                else
                    print_warning "Test coverage: $COVERAGE% (Consider improving)"
                fi
            fi
        fi
        
        return 0
    else
        print_error "Tests failed"
        print_warning "Check $RESULTS_DIR/pytest_check_$TIMESTAMP.txt for details"
        add_failed_check "tests"
        return 1
    fi
}

# Build check
check_build() {
    print_status "Checking if package builds correctly..."
    
    if poetry build > "$RESULTS_DIR/build_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Package build successful"
        return 0
    else
        print_error "Package build failed"
        print_warning "Check $RESULTS_DIR/build_check_$TIMESTAMP.txt for details"
        add_failed_check "build"
        return 1
    fi
}

# Docker build check
check_docker_build() {
    if [ -f "Dockerfile" ] && command -v docker &> /dev/null; then
        print_status "Checking Docker build..."
        
        if docker build -t llm-cost-tracker:ci-test . > "$RESULTS_DIR/docker_build_$TIMESTAMP.txt" 2>&1; then
            print_success "Docker build successful"
            
            # Clean up test image
            docker rmi llm-cost-tracker:ci-test > /dev/null 2>&1 || true
            
            return 0
        else
            print_error "Docker build failed"
            print_warning "Check $RESULTS_DIR/docker_build_$TIMESTAMP.txt for details"
            add_failed_check "docker-build"
            return 1
        fi
    else
        print_warning "Docker not available or no Dockerfile found, skipping Docker build"
        return 0
    fi
}

# Pre-commit validation
check_precommit() {
    print_status "Running pre-commit validation..."
    
    if poetry run pre-commit run --all-files > "$RESULTS_DIR/precommit_check_$TIMESTAMP.txt" 2>&1; then
        print_success "Pre-commit validation passed"
        return 0
    else
        print_error "Pre-commit validation failed"
        print_warning "Check $RESULTS_DIR/precommit_check_$TIMESTAMP.txt for details"
        add_failed_check "pre-commit"
        return 1
    fi
}

# Generate summary report
generate_summary() {
    print_status "Generating CI validation summary..."
    
    local summary_file="$RESULTS_DIR/ci_validation_summary_$TIMESTAMP.txt"
    
    cat > "$summary_file" << EOF
# CI/CD Pipeline Validation Summary
Generated: $(date)
Project: LLM Cost Tracker
Validation ID: $TIMESTAMP

## Checks Performed
EOF

    local total_checks=0
    local passed_checks=0
    
    # List all performed checks
    local checks=(
        "Prerequisites"
        "Dependencies Installation"
        "Code Formatting"
        "Import Sorting"
        "Linting"
        "Type Checking"
        "Security Checks"
        "Test Suite"
        "Package Build"
        "Docker Build"
        "Pre-commit Validation"
    )
    
    for check in "${checks[@]}"; do
        echo "- $check" >> "$summary_file"
        ((total_checks++))
    done
    
    echo "" >> "$summary_file"
    echo "## Results" >> "$summary_file"
    
    if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
        echo "✅ All checks passed successfully!" >> "$summary_file"
        passed_checks=$total_checks
    else
        echo "❌ ${#FAILED_CHECKS[@]} check(s) failed:" >> "$summary_file"
        for failed_check in "${FAILED_CHECKS[@]}"; do
            echo "  - $failed_check" >> "$summary_file"
        done
        passed_checks=$((total_checks - ${#FAILED_CHECKS[@]}))
    fi
    
    echo "" >> "$summary_file"
    echo "## Summary Statistics" >> "$summary_file"
    echo "- Total checks: $total_checks" >> "$summary_file"
    echo "- Passed: $passed_checks" >> "$summary_file"
    echo "- Failed: ${#FAILED_CHECKS[@]}" >> "$summary_file"
    echo "- Success rate: $((passed_checks * 100 / total_checks))%" >> "$summary_file"
    
    echo "" >> "$summary_file"
    echo "## Detailed Results" >> "$summary_file"
    echo "All detailed results are available in the $RESULTS_DIR directory." >> "$summary_file"
    
    print_success "Summary generated: $summary_file"
}

# Print final results
print_results() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}          CI Validation Results${NC}"
    echo -e "${BLUE}============================================${NC}\n"
    
    if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
        echo -e "${GREEN}🎉 All CI checks passed!${NC}"
        echo -e "${GREEN}Your code is ready for commit and CI/CD pipeline.${NC}"
        exit 0
    else
        echo -e "${RED}❌ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
        for failed_check in "${FAILED_CHECKS[@]}"; do
            echo -e "${RED}  - $failed_check${NC}"
        done
        
        echo -e "\n${YELLOW}Please fix the issues above before committing.${NC}"
        echo -e "${YELLOW}Detailed results are in: $RESULTS_DIR/${NC}"
        
        echo -e "\n${BLUE}Quick fixes:${NC}"
        echo "  Code formatting: poetry run black src tests"
        echo "  Import sorting: poetry run isort src tests"
        echo "  Run tests: poetry run pytest tests/"
        echo "  Security check: poetry run bandit -r src/"
        
        exit 1
    fi
}

# Main execution
main() {
    print_header
    
    check_prerequisites
    install_dependencies
    
    echo -e "\n${BLUE}Running CI validation checks...${NC}\n"
    
    # Run all checks (continue even if some fail)
    check_formatting || true
    check_import_sorting || true
    check_linting || true
    check_type_checking || true
    check_security || true
    run_tests || true
    check_build || true
    check_docker_build || true
    check_precommit || true
    
    echo -e "\n${BLUE}Generating results...${NC}\n"
    generate_summary
    
    print_results
}

# Handle script interruption
cleanup() {
    echo -e "\n${YELLOW}CI validation interrupted${NC}"
    exit 130
}

trap cleanup INT

# Run main function
main "$@"