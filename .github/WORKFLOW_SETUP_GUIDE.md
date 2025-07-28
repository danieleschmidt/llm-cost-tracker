# 🚀 GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up the comprehensive CI/CD workflows for the LLM Cost Tracker project.

## 📋 Prerequisites

Before setting up the workflows, ensure you have:

1. **Repository Permissions**: `workflows` permission to create/modify GitHub Actions
2. **Secrets Configuration**: Required API keys and configuration values
3. **Branch Protection**: Main branch protection rules configured
4. **Registry Access**: Docker registry credentials (if using external registry)

## 🔧 Workflow Templates Available

The following workflow templates are available in `.github/workflow-templates/`:

| Template | Purpose | Trigger |
|----------|---------|---------|
| `main-ci.yml` | Continuous Integration for main branch | Push to main |
| `pr-validation.yml` | Pull request validation pipeline | Pull request events |
| `security-scan.yml` | Security scanning and vulnerability assessment | Schedule + manual |
| `release.yml` | Automated release and deployment | Release creation |
| `semantic-release.yml` | Semantic versioning and changelog generation | Push to main |

## 🛠️ Setup Instructions

### Step 1: Configure Repository Secrets

Add the following secrets to your repository (`Settings` → `Secrets and variables` → `Actions`):

#### Required Secrets
```bash
# Docker Registry (if using external registry)
DOCKER_REGISTRY_URL=your-registry.com
DOCKER_REGISTRY_USERNAME=your-username
DOCKER_REGISTRY_PASSWORD=your-password

# Package Registry (for publishing)
PYPI_TOKEN=your-pypi-token
NPM_TOKEN=your-npm-token  # If applicable

# Notification Services
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Security Scanning
SNYK_TOKEN=your-snyk-token
CODECOV_TOKEN=your-codecov-token

# Cloud Deployments (if applicable)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
GOOGLE_APPLICATION_CREDENTIALS=your-gcp-credentials
AZURE_CLIENT_ID=your-azure-client-id
AZURE_CLIENT_SECRET=your-azure-secret
AZURE_TENANT_ID=your-azure-tenant

# Database (for integration tests)
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test_db
```

#### Optional Secrets
```bash
# Enhanced Security
GITHUB_TOKEN=ghp_your-github-token  # Usually auto-provided
SONARQUBE_TOKEN=your-sonar-token
WHITESOURCE_API_KEY=your-whitesource-key

# Monitoring & APM
DATADOG_API_KEY=your-datadog-key
NEW_RELIC_LICENSE_KEY=your-newrelic-key
SENTRY_DSN=your-sentry-dsn

# Custom Deployment
DEPLOYMENT_SSH_KEY=your-ssh-private-key
DEPLOYMENT_HOST=your-deployment-host
DEPLOYMENT_USER=your-deployment-user
```

### Step 2: Copy Workflow Templates

Copy the workflow templates from `.github/workflow-templates/` to `.github/workflows/`:

```bash
# From repository root
mkdir -p .github/workflows

# Copy all templates
cp .github/workflow-templates/main-ci.yml .github/workflows/
cp .github/workflow-templates/pr-validation.yml .github/workflows/
cp .github/workflow-templates/security-scan.yml .github/workflows/
cp .github/workflow-templates/release.yml .github/workflows/
cp .github/workflow-templates/semantic-release.yml .github/workflows/
```

### Step 3: Configure Branch Protection Rules

Set up branch protection for the main branch:

1. Go to `Settings` → `Branches`
2. Add rule for `main` branch with:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (minimum 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require signed commits (recommended)
   - ✅ Include administrators

#### Required Status Checks
Add these status checks to branch protection:
- `CI / Quality Checks`
- `CI / Security Scan`
- `CI / Unit Tests`
- `CI / Integration Tests`
- `CI / Build Verification`
- `CI / Container Security`
- `CI / Performance Tests`

### Step 4: Environment Configuration

Create deployment environments (`Settings` → `Environments`):

#### Development Environment
- **Name**: `development`
- **Protection Rules**: None
- **Environment Secrets**: Development-specific secrets

#### Staging Environment
- **Name**: `staging`
- **Protection Rules**: 
  - Required reviewers: 1
  - Wait timer: 0 minutes
- **Environment Secrets**: Staging-specific secrets

#### Production Environment
- **Name**: `production`
- **Protection Rules**:
  - Required reviewers: 2
  - Wait timer: 5 minutes
  - Protected branches: `main`
- **Environment Secrets**: Production-specific secrets

### Step 5: Workflow Customization

#### For main-ci.yml
Update the following variables in the workflow:
```yaml
env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"  # If using Node.js features
  REGISTRY: ghcr.io   # Change to your registry
  IMAGE_NAME: ${{ github.repository }}
```

#### For pr-validation.yml
Configure test matrix if needed:
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11"]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

#### For security-scan.yml
Configure security scanning schedule:
```yaml
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:      # Manual trigger
```

#### For release.yml
Configure release triggers:
```yaml
on:
  release:
    types: [published]
  push:
    tags:
      - 'v*'
```

### Step 6: Validate Setup

1. **Create a test branch**: `git checkout -b test-workflows`
2. **Make a small change**: Update a comment or documentation
3. **Create a pull request**: This should trigger `pr-validation.yml`
4. **Check workflow runs**: Go to `Actions` tab to verify workflows execute
5. **Review logs**: Ensure all steps complete successfully

## 🔍 Workflow Details

### Main CI Workflow (`main-ci.yml`)
**Triggers**: Push to main branch
**Jobs**:
1. **Code Quality**: Linting, formatting, type checking
2. **Security Scan**: SAST, dependency scanning, secrets detection
3. **Unit Tests**: Fast unit test execution with coverage
4. **Integration Tests**: Database and service integration tests
5. **Build**: Docker image building and verification
6. **Package**: Python package building and publishing
7. **Deploy to Staging**: Automatic staging deployment

### PR Validation Workflow (`pr-validation.yml`)
**Triggers**: Pull request opened, updated, or synchronized
**Jobs**:
1. **Quality Gates**: Code formatting and linting validation
2. **Security Checks**: Basic security scanning for PRs
3. **Test Matrix**: Cross-platform and cross-version testing
4. **Build Verification**: Ensure changes don't break builds
5. **Performance Check**: Benchmark regression detection
6. **Documentation**: Check for documentation updates
7. **Coverage Report**: Test coverage analysis and reporting

### Security Scan Workflow (`security-scan.yml`)
**Triggers**: Weekly schedule + manual dispatch
**Jobs**:
1. **SAST Scan**: Static application security testing
2. **Dependency Audit**: Check for vulnerable dependencies
3. **Container Scan**: Docker image vulnerability assessment
4. **Secrets Scan**: Repository-wide secrets detection
5. **OWASP Check**: Security best practices validation
6. **Compliance Report**: Generate security compliance report

### Release Workflow (`release.yml`)
**Triggers**: Release publication
**Jobs**:
1. **Pre-release Tests**: Final validation before release
2. **Build Production Images**: Multi-architecture builds
3. **Security Scan**: Final security validation
4. **Package Release**: Publish to PyPI, Docker Hub, etc.
5. **Deploy Production**: Blue-green production deployment
6. **Post-deploy Tests**: Smoke tests and health checks
7. **Notifications**: Success/failure notifications

### Semantic Release Workflow (`semantic-release.yml`)
**Triggers**: Push to main (after PR merge)
**Jobs**:
1. **Analyze Commits**: Parse conventional commits
2. **Determine Version**: Calculate semantic version bump
3. **Generate Changelog**: Create/update CHANGELOG.md
4. **Create Release**: GitHub release with artifacts
5. **Update Documentation**: Version-bump documentation
6. **Trigger Deployment**: Kick off release workflow

## 📊 Monitoring Workflow Health

### Workflow Metrics to Track
- **Success Rate**: Percentage of successful workflow runs
- **Execution Time**: Average time for each workflow type
- **Failure Patterns**: Common failure points and causes
- **Resource Usage**: Compute minutes consumed
- **Coverage Trends**: Test coverage over time

### Dashboard Creation
Create a workflow monitoring dashboard using:
```yaml
# .github/workflows/metrics.yml
name: Workflow Metrics
on:
  schedule:
    - cron: '0 8 * * 1'  # Weekly metrics report
  workflow_dispatch:

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - name: Generate Metrics Report
        uses: actions/github-script@v6
        with:
          script: |
            // Generate workflow metrics and send to monitoring
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Workflow Not Triggering
**Symptoms**: Workflows don't run on expected events
**Solutions**:
- Check branch protection rules
- Verify workflow file syntax (use YAML validator)
- Ensure workflows are in `.github/workflows/` directory
- Check repository permissions

#### 2. Secret Access Errors
**Symptoms**: Jobs fail with authentication errors
**Solutions**:
- Verify secret names match workflow references
- Check secret values don't contain special characters
- Ensure secrets are available in the correct environment
- Test with workflow_dispatch for debugging

#### 3. Test Failures in CI
**Symptoms**: Tests pass locally but fail in CI
**Solutions**:
- Check for environment-specific differences
- Verify test dependencies are installed
- Review test data and fixtures
- Check for race conditions in tests

#### 4. Build Timeouts
**Symptoms**: Workflows timeout or run too long
**Solutions**:
- Optimize Docker builds with multi-stage caching
- Use matrix builds to parallelize work
- Cache dependencies between runs
- Consider self-hosted runners for resource-intensive builds

#### 5. Deployment Failures
**Symptoms**: Deployment steps fail inconsistently
**Solutions**:
- Implement proper health checks
- Add retry logic for transient failures
- Use blue-green deployment for zero downtime
- Implement proper rollback mechanisms

### Debug Workflow
Use this debug workflow for troubleshooting:

```yaml
# .github/workflows/debug.yml
name: Debug Workflow
on:
  workflow_dispatch:
    inputs:
      debug_level:
        description: 'Debug level (info, debug, trace)'
        required: true
        default: 'info'

jobs:
  debug:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Debug Environment
        run: |
          echo "Event: ${{ github.event_name }}"
          echo "Ref: ${{ github.ref }}"
          echo "SHA: ${{ github.sha }}"
          echo "Actor: ${{ github.actor }}"
          env | grep GITHUB_ | sort
      
      - name: Debug Secrets
        run: |
          echo "Checking secret availability..."
          [ -n "${{ secrets.DOCKER_REGISTRY_PASSWORD }}" ] && echo "✅ DOCKER_REGISTRY_PASSWORD available" || echo "❌ DOCKER_REGISTRY_PASSWORD missing"
          [ -n "${{ secrets.PYPI_TOKEN }}" ] && echo "✅ PYPI_TOKEN available" || echo "❌ PYPI_TOKEN missing"
```

## 🎯 Best Practices

### 1. Security Best Practices
- Never log secrets or sensitive information
- Use environment-specific secrets
- Implement least-privilege access
- Regularly rotate credentials
- Use signed commits when possible

### 2. Performance Optimization
- Cache dependencies between runs
- Use matrix builds for parallelization
- Optimize Docker builds with layer caching
- Consider self-hosted runners for heavy workloads

### 3. Reliability Patterns
- Implement proper error handling
- Add retry logic for transient failures
- Use health checks for deployments
- Implement graceful rollback mechanisms

### 4. Monitoring & Alerting
- Set up failure notifications
- Monitor workflow execution times
- Track success/failure rates
- Alert on security scan findings

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Deployment Best Practices](https://docs.github.com/en/actions/deployment/about-deployments)

---

*This guide is part of the comprehensive SDLC implementation for the LLM Cost Tracker project. Update this document when workflow configurations change.*