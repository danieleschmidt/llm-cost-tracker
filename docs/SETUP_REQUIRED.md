# Manual Setup Requirements

This document lists items that require manual setup due to permission limitations.

## GitHub Repository Settings

### Security Configuration
1. Navigate to Settings → Security & analysis
2. Enable "Dependency graph"
3. Enable "Dependabot alerts"
4. Enable "Dependabot security updates"
5. Enable "Secret scanning"
6. Enable "Secret scanning push protection"

### Repository Details
- **Topics**: Add `python`, `opentelemetry`, `cost-tracking`, `llm`
- **Description**: "Self-hostable OpenTelemetry collector for LLM cost tracking"
- **Homepage**: Link to documentation or demo site
- **License**: Already configured (MIT)

### Branch Protection Rules

For the `main` branch, configure:
```
✓ Require a pull request before merging
  ✓ Require approvals (minimum: 1)
  ✓ Dismiss stale reviews when new commits are pushed
✓ Require status checks to pass before merging
  ✓ Require branches to be up to date before merging
✓ Restrict pushes that create files larger than 100 MB
✓ Do not allow bypassing the above settings
```

## GitHub Actions Workflows

Create the following workflow files manually:

### 1. `.github/workflows/ci.yml`
Basic CI pipeline for testing and validation

### 2. `.github/workflows/security.yml`
Security scanning and vulnerability assessment

### 3. `.github/workflows/release.yml`
Automated release management (requires PyPI_TOKEN secret)

## Required Secrets

Add these secrets in Settings → Secrets and variables → Actions:
- `PYPI_TOKEN`: For automated package publishing to PyPI

## Monitoring Integration

Configure external monitoring tools:
- **CodeCov**: For test coverage reporting
- **Snyk**: For additional security scanning
- **Sentry**: For error monitoring (if applicable)

## Documentation

Set up GitHub Pages for documentation hosting:
1. Go to Settings → Pages
2. Select source: GitHub Actions
3. Configure custom domain if available

## External Integrations

Consider setting up:
- **Renovate**: For automated dependency updates
- **Code quality badges** in README.md
- **Status page** for service monitoring