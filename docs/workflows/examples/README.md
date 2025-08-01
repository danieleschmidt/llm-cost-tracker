# GitHub Actions Workflow Examples

This directory contains complete GitHub Actions workflow examples that can be manually created by repository administrators.

## Overview

Due to GitHub App permission limitations, workflows must be manually created. This directory provides complete, tested workflow files that can be copied directly to `.github/workflows/`.

## Available Workflows

### Core CI/CD Workflows
- [`ci.yml`](ci.yml) - Complete CI/CD pipeline for main branch
- [`pr-validation.yml`](pr-validation.yml) - Pull request validation and testing
- [`release.yml`](release.yml) - Automated semantic release pipeline

### Security Workflows
- [`security-scan.yml`](security-scan.yml) - Comprehensive security scanning
- [`dependency-scan.yml`](dependency-scan.yml) - Dependency vulnerability scanning
- [`secret-scanning.yml`](secret-scanning.yml) - Secret detection and validation

### Quality Assurance
- [`code-quality.yml`](code-quality.yml) - Linting, formatting, and type checking
- [`performance-test.yml`](performance-test.yml) - Performance benchmarking
- [`integration-test.yml`](integration-test.yml) - End-to-end integration testing

### Deployment Workflows
- [`deploy-staging.yml`](deploy-staging.yml) - Staging environment deployment
- [`deploy-production.yml`](deploy-production.yml) - Production deployment workflow
- [`rollback.yml`](rollback.yml) - Emergency rollback procedures

### Maintenance Workflows
- [`dependency-update.yml`](dependency-update.yml) - Automated dependency updates
- [`cleanup.yml`](cleanup.yml) - Repository maintenance and cleanup
- [`backup.yml`](backup.yml) - Data backup automation

## Quick Setup Instructions

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Files
```bash
# Core workflows (required)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/pr-validation.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/

# Optional workflows
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 3. Configure Repository Secrets
Navigate to Settings > Secrets and Variables > Actions and add:

#### Required Secrets
- `CODECOV_TOKEN` - For code coverage reporting
- `SLACK_WEBHOOK_URL` - For deployment notifications (optional)

#### Optional Secrets
- `PYPI_TOKEN` - For automated PyPI publishing
- `DOCKER_HUB_TOKEN` - For Docker Hub publishing
- `SONAR_TOKEN` - For SonarCloud integration

### 4. Configure Repository Variables
Add the following repository variables:

#### Environment URLs
- `STAGING_URL` - Staging environment URL
- `PRODUCTION_URL` - Production environment URL

#### Configuration
- `PYTHON_VERSION` - Python version (default: "3.11")
- `NODE_VERSION` - Node.js version (default: "18")

### 5. Set Up Branch Protection
Navigate to Settings > Branches and configure protection for `main`:

- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Restrict pushes that create files (admin only)

Required status checks:
- `test-and-quality`
- `security-scanning`
- `build-and-push`

## Workflow Features

### CI/CD Pipeline (`ci.yml`)
- **Parallel execution** - Tests, security, and quality checks run concurrently
- **Multi-platform builds** - Supports AMD64 and ARM64 architectures
- **Container scanning** - Vulnerability scanning with Trivy
- **SBOM generation** - Software Bill of Materials for supply chain security
- **Performance monitoring** - Benchmark tracking and regression detection
- **Automated deployment** - Staging deployment with smoke tests

### Security Scanning (`security-scan.yml`)
- **CodeQL analysis** - GitHub's semantic code analysis
- **Dependency scanning** - Vulnerability detection in dependencies
- **Container scanning** - Docker image security analysis
- **Secret detection** - Prevention of credential leaks
- **SARIF reporting** - Security findings in GitHub Security tab

### Release Automation (`release.yml`)
- **Semantic versioning** - Automated version bumping based on commits
- **Changelog generation** - Automatic release notes
- **Multi-target publishing** - PyPI, Docker Hub, GitHub Releases
- **Release validation** - Automated testing of release artifacts

## Customization Guide

### Environment-Specific Configuration

#### Development Environment
```yaml
# For development workflows
environment:
  name: development
  url: http://dev.example.com
```

#### Production Environment
```yaml
# For production workflows
environment:
  name: production
  url: https://prod.example.com
  protection_rules:
    required_reviewers: 2
    wait_timer: 5
```

### Custom Notifications

#### Slack Integration
```yaml
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        "text": "Deployment Status: ${{ job.status }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Deployment Result:* ${{ job.status }}\n*Branch:* ${{ github.ref }}\n*Commit:* ${{ github.sha }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

#### Microsoft Teams Integration
```yaml
- name: Notify Teams
  uses: skitionek/notify-microsoft-teams@master
  if: always()
  with:
    webhook_url: ${{ secrets.TEAMS_WEBHOOK_URL }}
    title: "LLM Cost Tracker Deployment"
    message: "Deployment ${{ job.status }} for commit ${{ github.sha }}"
```

### Performance Optimization

#### Dependency Caching
```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      .venv
    key: ${{ runner.os }}-python-${{ hashFiles('**/poetry.lock') }}
    restore-keys: |
      ${{ runner.os }}-python-
```

#### Docker Layer Caching
```yaml
- name: Build with BuildKit cache
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ${{ steps.meta.outputs.tags }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
    build-args: |
      BUILDKIT_INLINE_CACHE=1
```

## Monitoring and Observability

### Workflow Metrics
Each workflow includes monitoring capabilities:

#### Success/Failure Tracking
```yaml
- name: Record workflow metrics
  if: always()
  run: |
    curl -X POST "${{ secrets.METRICS_ENDPOINT }}" \
      -H "Content-Type: application/json" \
      -d '{
        "workflow": "${{ github.workflow }}",
        "status": "${{ job.status }}",
        "duration": "${{ github.event.workflow_run.duration }}",
        "commit": "${{ github.sha }}"
      }'
```

#### Performance Benchmarking
```yaml
- name: Store performance metrics
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark.json
    github-token: ${{ secrets.GITHUB_TOKEN }}
    auto-push: true
    alert-threshold: '200%'
    comment-on-alert: true
```

## Security Best Practices

### Secret Management
1. **Never hardcode secrets** in workflow files
2. **Use environment-specific secrets** for different deployment targets
3. **Rotate secrets regularly** and update workflows accordingly
4. **Use least-privilege tokens** with minimal required permissions

### Secure Docker Builds
```yaml
- name: Build secure image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ${{ steps.meta.outputs.tags }}
    provenance: true  # Generate provenance attestation
    sbom: true        # Generate SBOM
    platforms: linux/amd64,linux/arm64
```

### Dependency Verification
```yaml
- name: Verify dependencies
  run: |
    # Verify Poetry lock file
    poetry check
    
    # Audit dependencies for vulnerabilities
    poetry run safety check
    
    # Check for outdated dependencies
    poetry show --outdated
```

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
1. Check branch protection rules
2. Verify workflow syntax with `yamllint`
3. Ensure required secrets are configured
4. Check repository permissions

#### Test Failures
1. Review test logs in Actions tab
2. Check database service connectivity
3. Verify environment variables
4. Run tests locally to reproduce

#### Build Failures
1. Check Dockerfile syntax
2. Verify build context includes necessary files
3. Review Docker build logs
4. Test multi-platform builds locally

#### Deployment Issues
1. Verify deployment targets are accessible
2. Check deployment credentials
3. Review infrastructure status
4. Validate configuration files

### Debug Commands

#### Local Workflow Testing
```bash
# Install act for local testing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

# Run workflow locally
act -j test --secret-file .secrets
```

#### Workflow Validation
```bash
# Validate workflow syntax
yamllint .github/workflows/*.yml

# Check for security issues
semgrep --config=auto .github/workflows/
```

## Migration Guide

### From Existing CI/CD Systems

#### Jenkins Migration
1. Convert Jenkinsfile stages to GitHub Actions jobs
2. Migrate pipeline parameters to workflow inputs
3. Replace Jenkins plugins with GitHub Actions
4. Update deployment scripts for GitHub Actions environment

#### GitLab CI Migration
1. Convert `.gitlab-ci.yml` jobs to GitHub Actions jobs
2. Update image references and dependencies
3. Migrate GitLab variables to GitHub secrets
4. Adapt deployment strategies

## Maintenance

### Regular Updates
- **Monthly**: Review and update action versions
- **Quarterly**: Audit security configurations
- **Annually**: Review workflow efficiency and optimization opportunities

### Version Pinning
Always pin action versions for security and reproducibility:
```yaml
# Good - pinned to specific version
uses: actions/checkout@v4.1.0

# Better - pinned to commit SHA
uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608
```

This comprehensive workflow documentation ensures reliable, secure, and efficient CI/CD operations for the LLM Cost Tracker project.