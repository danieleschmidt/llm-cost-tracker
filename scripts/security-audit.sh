#!/bin/bash
# Security Audit Script for LLM Cost Tracker
# This script runs comprehensive security checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="llm-cost-tracker"
RESULTS_DIR="security-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}🔒 Starting comprehensive security audit for $PROJECT_NAME${NC}"
echo "Results will be saved to: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo "----------------------------------------"

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

# Check if poetry is available
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed. Please install Poetry first."
        exit 1
    fi
    print_success "Poetry found"
}

# Install dependencies if needed
install_dependencies() {
    print_status "Installing dependencies..."
    poetry install --with dev
    print_success "Dependencies installed"
}

# Run Bandit security scan
run_bandit() {
    print_status "Running Bandit security scan..."
    
    poetry run bandit -r src/ \
        --format json \
        --output "$RESULTS_DIR/bandit_report_$TIMESTAMP.json" \
        --severity-level low \
        --confidence-level low || true
    
    poetry run bandit -r src/ \
        --format txt \
        --output "$RESULTS_DIR/bandit_report_$TIMESTAMP.txt" \
        --severity-level medium || true
    
    print_success "Bandit scan completed"
}

# Run Safety dependency check
run_safety() {
    print_status "Running Safety dependency vulnerability check..."
    
    poetry run safety check \
        --json \
        --output "$RESULTS_DIR/safety_report_$TIMESTAMP.json" || true
    
    poetry run safety check \
        --output "$RESULTS_DIR/safety_report_$TIMESTAMP.txt" || true
    
    print_success "Safety check completed"
}

# Run Semgrep static analysis
run_semgrep() {
    if command -v semgrep &> /dev/null; then
        print_status "Running Semgrep static analysis..."
        
        semgrep --config=auto \
            --json \
            --output="$RESULTS_DIR/semgrep_report_$TIMESTAMP.json" \
            src/ || true
        
        semgrep --config=auto \
            --output="$RESULTS_DIR/semgrep_report_$TIMESTAMP.txt" \
            src/ || true
        
        print_success "Semgrep analysis completed"
    else
        print_warning "Semgrep not found, skipping static analysis"
    fi
}

# Check for secrets in code
check_secrets() {
    print_status "Checking for hardcoded secrets..."
    
    # Check for common secret patterns
    SECRET_PATTERNS=(
        "password\s*=\s*['\"][^'\"]*['\"]"
        "api[_-]?key\s*=\s*['\"][^'\"]*['\"]"
        "secret\s*=\s*['\"][^'\"]*['\"]"
        "token\s*=\s*['\"][^'\"]*['\"]"
        "-----BEGIN.*PRIVATE KEY-----"
        "sk_live_[0-9a-z]+"
        "pk_live_[0-9a-z]+"
    )
    
    echo "Scanning for hardcoded secrets..." > "$RESULTS_DIR/secrets_check_$TIMESTAMP.txt"
    
    for pattern in "${SECRET_PATTERNS[@]}"; do
        echo "Checking pattern: $pattern" >> "$RESULTS_DIR/secrets_check_$TIMESTAMP.txt"
        grep -r -i -E "$pattern" src/ >> "$RESULTS_DIR/secrets_check_$TIMESTAMP.txt" || true
        echo "---" >> "$RESULTS_DIR/secrets_check_$TIMESTAMP.txt"
    done
    
    print_success "Secret scan completed"
}

# Check file permissions
check_permissions() {
    print_status "Checking file permissions..."
    
    echo "File permission audit:" > "$RESULTS_DIR/permissions_check_$TIMESTAMP.txt"
    echo "Date: $(date)" >> "$RESULTS_DIR/permissions_check_$TIMESTAMP.txt"
    echo "---" >> "$RESULTS_DIR/permissions_check_$TIMESTAMP.txt"
    
    # Check for files with overly permissive permissions
    find . -type f -perm -002 -not -path "./.git/*" >> "$RESULTS_DIR/permissions_check_$TIMESTAMP.txt" || true
    
    # Check for executable files that shouldn't be
    find src/ -name "*.py" -executable >> "$RESULTS_DIR/permissions_check_$TIMESTAMP.txt" || true
    
    print_success "Permission check completed"
}

# Analyze Docker security
check_docker_security() {
    if [ -f "Dockerfile" ]; then
        print_status "Analyzing Docker security..."
        
        if command -v hadolint &> /dev/null; then
            hadolint Dockerfile > "$RESULTS_DIR/hadolint_report_$TIMESTAMP.txt" || true
            print_success "Hadolint analysis completed"
        else
            print_warning "Hadolint not found, skipping Dockerfile analysis"
        fi
        
        # Check for security best practices in Dockerfile
        echo "Docker Security Analysis:" > "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        echo "Date: $(date)" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        echo "---" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        
        # Check if running as non-root user
        if grep -q "USER" Dockerfile; then
            echo "✓ Non-root user specified" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        else
            echo "⚠ Running as root user" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        fi
        
        # Check for HEALTHCHECK
        if grep -q "HEALTHCHECK" Dockerfile; then
            echo "✓ Health check configured" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        else
            echo "⚠ No health check configured" >> "$RESULTS_DIR/docker_security_$TIMESTAMP.txt"
        fi
        
        print_success "Docker security analysis completed"
    else
        print_warning "No Dockerfile found, skipping Docker security check"
    fi
}

# Check environment configuration
check_env_config() {
    print_status "Checking environment configuration..."
    
    echo "Environment Configuration Audit:" > "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
    echo "Date: $(date)" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
    echo "---" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
    
    # Check .env.example for security best practices
    if [ -f ".env.example" ]; then
        echo "Checking .env.example for security issues..." >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
        
        # Check for default passwords
        if grep -i "password.*=.*password\|password.*=.*123\|password.*=.*admin" .env.example; then
            echo "⚠ Default passwords found in .env.example" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
        else
            echo "✓ No obvious default passwords in .env.example" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
        fi
        
        # Check for debug mode
        if grep -i "debug.*=.*true" .env.example; then
            echo "⚠ Debug mode enabled in .env.example" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
        else
            echo "✓ Debug mode not enabled in .env.example" >> "$RESULTS_DIR/env_config_$TIMESTAMP.txt"
        fi
    fi
    
    print_success "Environment configuration check completed"
}

# Generate security report summary
generate_summary() {
    print_status "Generating security audit summary..."
    
    SUMMARY_FILE="$RESULTS_DIR/security_audit_summary_$TIMESTAMP.txt"
    
    cat > "$SUMMARY_FILE" << EOF
# Security Audit Summary
Generated: $(date)
Project: $PROJECT_NAME
Scan ID: $TIMESTAMP

## Scans Performed
- Bandit static analysis
- Safety dependency vulnerability check
- Secret detection
- File permission analysis
- Docker security analysis
- Environment configuration check

## Results Location
All detailed results are available in the $RESULTS_DIR directory:

EOF

    # List all generated files
    ls -la "$RESULTS_DIR"/*"$TIMESTAMP"* >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

## Next Steps
1. Review all generated reports
2. Address any HIGH or CRITICAL severity issues
3. Update dependencies with known vulnerabilities
4. Fix any security misconfigurations
5. Re-run this audit after fixes

## Recommendations
- Run this audit regularly (weekly/monthly)
- Integrate security checks into CI/CD pipeline
- Keep dependencies updated
- Follow security best practices
- Conduct periodic penetration testing

EOF

    print_success "Security audit summary generated: $SUMMARY_FILE"
}

# Main execution
main() {
    check_poetry
    install_dependencies
    
    echo -e "\n${BLUE}Starting security scans...${NC}\n"
    
    run_bandit
    run_safety
    run_semgrep
    check_secrets
    check_permissions
    check_docker_security
    check_env_config
    
    echo -e "\n${BLUE}Generating summary...${NC}\n"
    generate_summary
    
    echo -e "\n${GREEN}🎉 Security audit completed successfully!${NC}"
    echo -e "📊 View the summary: ${YELLOW}$RESULTS_DIR/security_audit_summary_$TIMESTAMP.txt${NC}"
    echo -e "📁 All results in: ${YELLOW}$RESULTS_DIR/${NC}"
    
    # Check if there are any critical issues
    if [ -f "$RESULTS_DIR/bandit_report_$TIMESTAMP.json" ]; then
        CRITICAL_ISSUES=$(jq '.results[] | select(.issue_severity == "HIGH")' "$RESULTS_DIR/bandit_report_$TIMESTAMP.json" 2>/dev/null | wc -l || echo "0")
        if [ "$CRITICAL_ISSUES" -gt 0 ]; then
            print_warning "$CRITICAL_ISSUES high-severity issues found. Please review immediately!"
            exit 1
        fi
    fi
    
    print_success "No critical security issues detected"
}

# Handle script interruption
cleanup() {
    echo -e "\n${YELLOW}Security audit interrupted${NC}"
    exit 130
}

trap cleanup INT

# Run main function
main "$@"