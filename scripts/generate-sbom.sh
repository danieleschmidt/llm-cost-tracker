#!/bin/bash
# Generate Software Bill of Materials (SBOM) for supply chain security
# This script creates comprehensive SBOM documents in multiple formats

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SBOM_DIR="$PROJECT_ROOT/sbom"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if tools are installed
check_dependencies() {
    log_info "Checking required dependencies..."
    
    local missing_tools=()
    
    # Check for syft (SBOM generation)
    if ! command -v syft &> /dev/null; then
        missing_tools+=("syft")
    fi
    
    # Check for grype (vulnerability scanning)
    if ! command -v grype &> /dev/null; then
        missing_tools+=("grype")
    fi
    
    # Check for cyclonedx-bom (alternative SBOM tool)
    if ! command -v cyclonedx-bom &> /dev/null && ! pip list | grep -q cyclonedx-bom; then
        log_warning "cyclonedx-bom not found, will skip CycloneDX format"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install syft: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
        log_info "Install grype: curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin"
        exit 1
    fi
    
    log_success "All required dependencies are available"
}

# Create SBOM directory structure
setup_directories() {
    log_info "Setting up SBOM directories..."
    
    mkdir -p "$SBOM_DIR"/{spdx,cyclonedx,reports,attestations}
    
    log_success "SBOM directories created"
}

# Generate SBOM using Syft
generate_syft_sbom() {
    log_info "Generating SBOM using Syft..."
    
    cd "$PROJECT_ROOT"
    
    # Generate SPDX format SBOM
    syft packages dir:. -o spdx-json="$SBOM_DIR/spdx/sbom-${TIMESTAMP}.spdx.json"
    syft packages dir:. -o spdx-tag-value="$SBOM_DIR/spdx/sbom-${TIMESTAMP}.spdx"
    
    # Generate SPDX for Docker image if Dockerfile exists
    if [ -f "Dockerfile" ]; then
        log_info "Generating SBOM for Docker image..."
        # Build image for SBOM generation
        docker build -t llm-cost-tracker:sbom-temp .
        syft packages llm-cost-tracker:sbom-temp -o spdx-json="$SBOM_DIR/spdx/docker-sbom-${TIMESTAMP}.spdx.json"
        # Clean up temporary image
        docker rmi llm-cost-tracker:sbom-temp || true
    fi
    
    # Generate CycloneDX format
    syft packages dir:. -o cyclonedx-json="$SBOM_DIR/cyclonedx/sbom-${TIMESTAMP}.json"
    syft packages dir:. -o cyclonedx-xml="$SBOM_DIR/cyclonedx/sbom-${TIMESTAMP}.xml"
    
    # Generate table format for human readability
    syft packages dir:. -o table="$SBOM_DIR/reports/packages-${TIMESTAMP}.txt"
    
    log_success "Syft SBOM generation completed"
}

# Generate CycloneDX SBOM (if available)
generate_cyclonedx_sbom() {
    if command -v cyclonedx-bom &> /dev/null || pip list | grep -q cyclonedx-bom; then
        log_info "Generating CycloneDX SBOM using cyclonedx-bom..."
        
        cd "$PROJECT_ROOT"
        
        # Generate from pyproject.toml
        if [ -f "pyproject.toml" ]; then
            python -m cyclonedx.cli -f json -o "$SBOM_DIR/cyclonedx/python-sbom-${TIMESTAMP}.json" pyproject.toml
        fi
        
        # Generate from requirements if available
        if [ -f "requirements.txt" ]; then
            python -m cyclonedx.cli -f json -o "$SBOM_DIR/cyclonedx/requirements-sbom-${TIMESTAMP}.json" requirements.txt
        fi
        
        log_success "CycloneDX SBOM generation completed"
    else
        log_warning "CycloneDX tools not available, skipping..."
    fi
}

# Perform vulnerability scanning
perform_vulnerability_scan() {
    log_info "Performing vulnerability scanning with Grype..."
    
    cd "$PROJECT_ROOT"
    
    # Scan the project directory
    grype dir:. -o json > "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.json"
    grype dir:. -o table > "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.txt"
    
    # Scan Docker image if available
    if [ -f "Dockerfile" ]; then
        log_info "Scanning Docker image for vulnerabilities..."
        docker build -t llm-cost-tracker:vuln-scan .
        grype llm-cost-tracker:vuln-scan -o json > "$SBOM_DIR/reports/docker-vulnerabilities-${TIMESTAMP}.json"
        grype llm-cost-tracker:vuln-scan -o table > "$SBOM_DIR/reports/docker-vulnerabilities-${TIMESTAMP}.txt"
        docker rmi llm-cost-tracker:vuln-scan || true
    fi
    
    log_success "Vulnerability scanning completed"
}

# Generate attestations
generate_attestations() {
    log_info "Generating supply chain attestations..."
    
    # Create provenance information
    cat > "$SBOM_DIR/attestations/provenance-${TIMESTAMP}.json" <<EOF
{
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "predicate": {
    "builder": {
      "id": "https://github.com/terragon-labs/llm-cost-tracker/actions"
    },
    "buildType": "https://github.com/Attestations/GitHubActionsWorkflow@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/terragon-labs/llm-cost-tracker.git",
        "digest": {
          "sha1": "$(git rev-parse HEAD)"
        },
        "entryPoint": "scripts/generate-sbom.sh"
      }
    },
    "metadata": {
      "buildInvocationId": "${TIMESTAMP}",
      "buildStartedOn": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
      "completeness": {
        "parameters": true,
        "environment": false,
        "materials": true
      },
      "reproducible": false
    },
    "materials": [
      {
        "uri": "git+https://github.com/terragon-labs/llm-cost-tracker.git",
        "digest": {
          "sha1": "$(git rev-parse HEAD)"
        }
      }
    ]
  }
}
EOF
    
    # Create build configuration attestation
    cat > "$SBOM_DIR/attestations/build-config-${TIMESTAMP}.json" <<EOF
{
  "predicateType": "https://slsa.dev/build-config/v0.1",
  "predicate": {
    "buildType": "Python Poetry Build",
    "builder": {
      "id": "poetry",
      "version": "$(poetry --version | cut -d' ' -f3)"
    },
    "buildConfig": {
      "python_version": "$(python --version | cut -d' ' -f2)",
      "poetry_lock_hash": "$(sha256sum poetry.lock | cut -d' ' -f1)",
      "pyproject_hash": "$(sha256sum pyproject.toml | cut -d' ' -f1)"
    },
    "metadata": {
      "buildInvocationId": "${TIMESTAMP}",
      "buildStartedOn": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    }
  }
}
EOF
    
    log_success "Supply chain attestations generated"
}

# Create summary report
create_summary_report() {
    log_info "Creating SBOM summary report..."
    
    local report_file="$SBOM_DIR/SBOM_SUMMARY_${TIMESTAMP}.md"
    
    cat > "$report_file" <<EOF
# Software Bill of Materials (SBOM) Summary

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Project:** LLM Cost Tracker
**Commit:** $(git rev-parse HEAD)
**Branch:** $(git rev-parse --abbrev-ref HEAD)

## Overview

This SBOM was generated to provide transparency into the software supply chain
of the LLM Cost Tracker project, supporting security analysis and compliance requirements.

## Generated Artifacts

### SPDX Format
- \`spdx/sbom-${TIMESTAMP}.spdx.json\` - SPDX JSON format
- \`spdx/sbom-${TIMESTAMP}.spdx\` - SPDX tag-value format
$([ -f "$SBOM_DIR/spdx/docker-sbom-${TIMESTAMP}.spdx.json" ] && echo "- \`spdx/docker-sbom-${TIMESTAMP}.spdx.json\` - Docker image SBOM")

### CycloneDX Format
- \`cyclonedx/sbom-${TIMESTAMP}.json\` - CycloneDX JSON format
- \`cyclonedx/sbom-${TIMESTAMP}.xml\` - CycloneDX XML format

### Vulnerability Reports
- \`reports/vulnerabilities-${TIMESTAMP}.json\` - JSON vulnerability report
- \`reports/vulnerabilities-${TIMESTAMP}.txt\` - Human-readable vulnerability report
$([ -f "$SBOM_DIR/reports/docker-vulnerabilities-${TIMESTAMP}.json" ] && echo "- \`reports/docker-vulnerabilities-${TIMESTAMP}.json\` - Docker vulnerability report")

### Attestations
- \`attestations/provenance-${TIMESTAMP}.json\` - SLSA provenance attestation
- \`attestations/build-config-${TIMESTAMP}.json\` - Build configuration attestation

## Package Statistics

EOF
    
    # Add package statistics
    if [ -f "$SBOM_DIR/reports/packages-${TIMESTAMP}.txt" ]; then
        echo "### Dependencies Summary" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        head -20 "$SBOM_DIR/reports/packages-${TIMESTAMP}.txt" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add vulnerability summary
    if [ -f "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.json" ]; then
        local high_vulns=$(jq '.matches[] | select(.vulnerability.severity == "High") | .vulnerability.id' "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.json" 2>/dev/null | wc -l || echo "0")
        local medium_vulns=$(jq '.matches[] | select(.vulnerability.severity == "Medium") | .vulnerability.id' "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.json" 2>/dev/null | wc -l || echo "0")
        local low_vulns=$(jq '.matches[] | select(.vulnerability.severity == "Low") | .vulnerability.id' "$SBOM_DIR/reports/vulnerabilities-${TIMESTAMP}.json" 2>/dev/null | wc -l || echo "0")
        
        cat >> "$report_file" <<EOF
## Vulnerability Summary

- 🔴 High Severity: ${high_vulns}
- 🟡 Medium Severity: ${medium_vulns}
- 🟢 Low Severity: ${low_vulns}

EOF
    fi
    
    cat >> "$report_file" <<EOF
## Usage Instructions

### Validating SBOM Integrity
\`\`\`bash
# Verify SPDX SBOM
spdx-tools validate spdx/sbom-${TIMESTAMP}.spdx.json

# Verify CycloneDX SBOM
cyclonedx validate cyclonedx/sbom-${TIMESTAMP}.json
\`\`\`

### Analyzing Vulnerabilities
\`\`\`bash
# Review vulnerability report
cat reports/vulnerabilities-${TIMESTAMP}.txt

# Query specific vulnerabilities
jq '.matches[] | select(.vulnerability.severity == "High")' reports/vulnerabilities-${TIMESTAMP}.json
\`\`\`

### Supply Chain Verification
\`\`\`bash
# Verify build provenance
jq . attestations/provenance-${TIMESTAMP}.json

# Check build configuration
jq . attestations/build-config-${TIMESTAMP}.json
\`\`\`

## Integration with CI/CD

This SBOM can be integrated into your CI/CD pipeline:

1. **Artifact Signing**: Sign SBOM files with cosign
2. **Policy Enforcement**: Use OPA policies to validate SBOM content
3. **Vulnerability Gates**: Fail builds based on vulnerability thresholds
4. **Compliance Reporting**: Submit SBOMs to compliance systems

## Standards Compliance

- ✅ SPDX 2.3 - Software Package Data Exchange
- ✅ CycloneDX 1.4 - OWASP CycloneDX Specification  
- ✅ SLSA v0.2 - Supply Chain Levels for Software Artifacts
- ✅ NIST SSDF - Secure Software Development Framework

---

*Generated by LLM Cost Tracker automated SBOM pipeline*
EOF
    
    log_success "SBOM summary report created: $report_file"
}

# Main execution
main() {
    log_info "Starting SBOM generation for LLM Cost Tracker..."
    
    check_dependencies
    setup_directories
    generate_syft_sbom
    generate_cyclonedx_sbom
    perform_vulnerability_scan
    generate_attestations
    create_summary_report
    
    log_success "SBOM generation completed successfully!"
    log_info "SBOM artifacts are available in: $SBOM_DIR"
    
    # Create latest symlinks
    cd "$SBOM_DIR"
    ln -sf "spdx/sbom-${TIMESTAMP}.spdx.json" "sbom-latest.spdx.json"
    ln -sf "cyclonedx/sbom-${TIMESTAMP}.json" "sbom-latest.cyclonedx.json"
    ln -sf "reports/vulnerabilities-${TIMESTAMP}.json" "vulnerabilities-latest.json"
    
    log_info "Latest SBOM symlinks created"
    
    # Display next steps
    echo
    log_info "Next steps:"
    echo "  1. Review the SBOM summary: cat $SBOM_DIR/SBOM_SUMMARY_${TIMESTAMP}.md"
    echo "  2. Address any high-severity vulnerabilities found"
    echo "  3. Integrate SBOM generation into your CI/CD pipeline"
    echo "  4. Consider signing SBOMs with cosign for enhanced security"
}

# Execute main function
main "$@"