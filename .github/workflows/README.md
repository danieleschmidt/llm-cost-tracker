# GitHub Actions Workflows

This directory contains GitHub Actions workflow templates for the LLM Cost Tracker project. These workflows implement a comprehensive CI/CD pipeline with security, testing, and deployment automation.

> **Note**: Due to security requirements, these workflows need to be manually reviewed and created by repository administrators with appropriate permissions.

## Workflow Overview

### 1. Pull Request Validation (`pr-validation.yml`)
- **Trigger**: Pull requests to main/develop branches
- **Purpose**: Validate code quality, run tests, and perform security checks
- **Jobs**:
  - Code quality checks (linting, formatting, type checking)
  - Unit and integration tests with coverage reporting
  - Security scanning (SAST, dependency scanning, secrets detection)
  - Build verification
  - Performance regression testing

### 2. Main Branch CI (`main-ci.yml`)
- **Trigger**: Push to main branch
- **Purpose**: Full CI pipeline with deployment to staging
- **Jobs**:
  - All PR validation checks
  - Docker image building and scanning
  - Deployment to staging environment
  - End-to-end testing
  - Security compliance verification

### 3. Release Pipeline (`release.yml`)
- **Trigger**: Git tags matching `v*` pattern
- **Purpose**: Automated release process
- **Jobs**:
  - Version validation and changelog generation
  - Production build with multi-arch support
  - Security scanning and SBOM generation
  - Container signing with Cosign
  - Deployment to production
  - Release notes generation and GitHub release creation

### 4. Security Scanning (`security.yml`)
- **Trigger**: Schedule (daily), manual dispatch
- **Purpose**: Comprehensive security scanning
- **Jobs**:
  - SAST with CodeQL
  - Dependency vulnerability scanning
  - Container security scanning
  - Infrastructure security checks
  - Compliance reporting

### 5. Dependency Updates (`dependency-updates.yml`)
- **Trigger**: Schedule (weekly), Dependabot alerts
- **Purpose**: Automated dependency management
- **Jobs**:
  - Security-first dependency updates
  - Automated testing of updates
  - Pull request creation for manual review
  - Risk assessment and prioritization

### 6. Performance Monitoring (`performance.yml`)
- **Trigger**: Schedule (daily), performance events
- **Purpose**: Continuous performance monitoring
- **Jobs**:
  - Performance benchmarking
  - Load testing
  - Resource usage monitoring
  - Performance regression detection

## Workflow Features

### Security
- **SAST**: CodeQL static analysis
- **SCA**: OWASP Dependency-Check with NIST NVD
- **Container Security**: Trivy vulnerability scanning
- **Secrets Detection**: TruffleHog integration
- **Supply Chain Security**: SLSA attestation and container signing
- **Compliance**: FIPS 140-2, SOC 2 Type II alignment

### Quality Gates
- **Code Coverage**: Minimum 80% requirement
- **Performance**: Latency and throughput thresholds
- **Security**: Zero critical vulnerabilities policy
- **Documentation**: API documentation freshness
- **Accessibility**: WCAG 2.1 AA compliance (for web components)

### Deployment Strategy
- **Blue-Green Deployment**: Zero-downtime releases
- **Canary Releases**: Gradual rollout with monitoring
- **Feature Flags**: A/B testing and rollback capabilities
- **Health Checks**: Comprehensive monitoring and alerting
- **Rollback**: Automated rollback on failure detection

### Notifications
- **Slack Integration**: Build status and alerts
- **Email Notifications**: Critical failures and releases
- **GitHub Status Checks**: Required for branch protection
- **Metrics Export**: Prometheus metrics for monitoring

## Environment Configuration

### Secrets Required
```yaml
# Repository secrets that need to be configured:
DOCKER_REGISTRY_TOKEN     # Container registry authentication
SLACK_WEBHOOK_URL         # Slack notifications
SONAR_TOKEN              # SonarCloud integration
SNYK_TOKEN               # Snyk security scanning
COSIGN_PRIVATE_KEY       # Container signing
DEPLOYMENT_TOKEN         # Production deployment
GRAFANA_API_KEY          # Monitoring integration
```

### Variables Required
```yaml
# Repository variables that need to be configured:
DOCKER_REGISTRY          # Container registry URL
STAGING_URL              # Staging environment URL
PRODUCTION_URL           # Production environment URL
SLACK_CHANNEL           # Notification channel
SONAR_PROJECT_KEY       # SonarCloud project
```

## Branch Protection Rules

### Main Branch
- Require pull request reviews (2 reviewers minimum)
- Require status checks to pass:
  - `ci/code-quality`
  - `ci/tests`
  - `ci/security`
  - `ci/build`
- Require up-to-date branches
- Require conversation resolution
- Restrict pushes to administrators

### Develop Branch
- Require pull request reviews (1 reviewer minimum)
- Require status checks to pass:
  - `ci/code-quality`
  - `ci/tests`
  - `ci/build`
- Allow force pushes for administrators

## Monitoring and Observability

### DORA Metrics
- **Deployment Frequency**: Tracked via deployment events
- **Lead Time for Changes**: Calculated from commit to deployment
- **Change Failure Rate**: Monitored via rollback events
- **Mean Time to Recovery**: Measured from incident to resolution

### Performance Metrics
- **Build Duration**: CI/CD pipeline execution time
- **Test Coverage**: Code coverage percentage
- **Security Score**: Vulnerability count and severity
- **Quality Score**: Code quality metrics from SonarCloud

## Usage Instructions

### Setting Up Workflows

1. **Review Templates**: Examine workflow templates in this directory
2. **Configure Secrets**: Add required secrets to repository settings
3. **Set Variables**: Configure repository variables
4. **Create Workflows**: Copy templates to `.github/workflows/` directory
5. **Test Workflows**: Trigger workflows manually to verify setup
6. **Configure Branch Protection**: Apply protection rules to main/develop branches

### Local Development

```bash
# Install act for local workflow testing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test workflows locally
act -j test                    # Run test job
act pull_request              # Simulate PR event
act push -e push.json         # Test push event
```

### Workflow Debugging

```bash
# Enable debug logging
gh workflow run ci.yml --ref main -f debug=true

# View workflow logs
gh run list
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

## Best Practices

### Workflow Design
- **Fail Fast**: Run quick checks first
- **Parallel Execution**: Maximize concurrency
- **Artifact Caching**: Cache dependencies and build outputs
- **Resource Optimization**: Use appropriate runner sizes
- **Security by Default**: Enable security scanning by default

### Secret Management
- **Least Privilege**: Grant minimum required permissions
- **Rotation**: Regular secret rotation schedule
- **Monitoring**: Audit secret usage
- **Scoping**: Environment-specific secrets

### Performance Optimization
- **Matrix Builds**: Test across multiple environments
- **Conditional Execution**: Skip unnecessary jobs
- **Caching Strategy**: Cache Poetry dependencies and Docker layers
- **Artifact Management**: Cleanup old artifacts

## Troubleshooting

### Common Issues

1. **Secret Access**: Verify secret names and scoping
2. **Permission Errors**: Check GITHUB_TOKEN permissions
3. **Build Failures**: Review dependency conflicts
4. **Timeout Issues**: Adjust timeout values for long-running jobs
5. **Cache Misses**: Verify cache key patterns

### Support

- **Documentation**: GitHub Actions documentation
- **Community**: GitHub Community Discussions
- **Support**: GitHub Support for enterprise accounts

## Compliance and Governance

### Audit Trail
- All workflow runs are logged and retained
- Deployment approvals are tracked
- Security scan results are archived
- Compliance reports are generated monthly

### Change Management
- Workflow changes require review
- Production deployments require approval
- Emergency procedures are documented
- Rollback procedures are tested

---

*This CI/CD pipeline implements industry best practices for security, quality, and reliability while maintaining developer productivity and rapid delivery capabilities.*