# Manual GitHub Actions Setup Instructions

This guide provides step-by-step instructions for manually setting up GitHub Actions workflows due to GitHub App permission limitations.

## Prerequisites

- Repository admin access
- GitHub Pro/Team/Enterprise account (for some features)
- Access to organization secrets management

## Step 1: Create Workflow Directory Structure

```bash
# In your repository root
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE
```

## Step 2: Copy Workflow Files

Navigate to the `docs/workflows/examples/` directory in this repository and copy the following files to `.github/workflows/`:

### Required Workflows (Copy these first)
```bash
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/pr-validation.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
```

### Optional Workflows (Copy as needed)
```bash
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/performance-test.yml .github/workflows/
```

## Step 3: Configure Repository Secrets

Navigate to **Settings > Secrets and variables > Actions** and add the following secrets:

### Required Secrets

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `CODECOV_TOKEN` | Code coverage reporting | Get from [codecov.io](https://codecov.io) after linking repository |

### Optional Secrets (for enhanced features)

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `SLACK_WEBHOOK_URL` | Slack notifications | Create webhook in Slack workspace settings |
| `TEAMS_WEBHOOK_URL` | Microsoft Teams notifications | Create webhook in Teams channel settings |
| `PYPI_TOKEN` | PyPI package publishing | Generate API token at [pypi.org](https://pypi.org/manage/account/token/) |
| `DOCKER_HUB_TOKEN` | Docker Hub publishing | Generate access token in Docker Hub settings |
| `SONAR_TOKEN` | SonarCloud integration | Create token at [sonarcloud.io](https://sonarcloud.io) |

### How to Add Secrets

1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Click **New repository secret**
3. Enter the secret name and value
4. Click **Add secret**

## Step 4: Configure Repository Variables

Navigate to **Settings > Secrets and variables > Actions > Variables** and add:

### Environment Configuration

| Variable Name | Default Value | Description |
|---------------|---------------|-------------|
| `PYTHON_VERSION` | `"3.11"` | Python version for CI/CD |
| `NODE_VERSION` | `"18"` | Node.js version (if needed) |
| `POETRY_VERSION` | `"1.7.1"` | Poetry version |

### Environment URLs

| Variable Name | Example Value | Description |
|---------------|---------------|-------------|
| `STAGING_URL` | `https://staging.llm-cost-tracker.com` | Staging environment URL |
| `PRODUCTION_URL` | `https://llm-cost-tracker.com` | Production environment URL |

## Step 5: Set Up Branch Protection Rules

1. Navigate to **Settings > Branches**
2. Click **Add rule** or edit existing rule for `main` branch
3. Configure the following settings:

### Required Settings
- ✅ **Require pull request reviews before merging**
  - Required approving reviews: `1` (or `2` for production)
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if CODEOWNERS file exists)

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Add these required status checks:
    - `test-and-quality`
    - `security-scanning` 
    - `build-and-push`
    - `pr-validation`

- ✅ **Restrict pushes that create files**
  - Add administrators who can bypass (usually just you)

### Optional Settings
- ✅ **Require signed commits** (recommended for security)
- ✅ **Include administrators** (enforce rules for admins too)
- ✅ **Allow force pushes** (only for emergency situations)

## Step 6: Configure Repository Settings

### General Settings

1. Navigate to **Settings > General**
2. Configure these settings:

**Repository visibility**: Ensure appropriate visibility (Private/Public)

**Features**: Enable/disable as needed:
- ✅ Wikis (for additional documentation)
- ✅ Issues (for bug tracking)
- ✅ Sponsorships (if open source)
- ✅ Preserve this repository (for important projects)
- ✅ Discussions (for community engagement)

**Pull Requests**:
- ✅ Allow merge commits
- ✅ Allow squash merging (recommended for clean history)
- ✅ Allow rebase merging
- ✅ Always suggest updating pull request branches
- ✅ Allow auto-merge
- ✅ Automatically delete head branches

### Security Settings

1. Navigate to **Settings > Security & analysis**
2. Enable these features:

**Dependency graph**: ✅ Enable

**Dependabot alerts**: ✅ Enable
- Configure alert recipients
- Set up auto-dismiss rules for false positives

**Dependabot security updates**: ✅ Enable

**Code scanning**: ✅ Enable
- Select "GitHub CodeQL" for automated analysis
- Configure additional tools if needed

**Secret scanning**: ✅ Enable (available on public repos and private repos with GitHub Advanced Security)

### Access Settings

1. Navigate to **Settings > Manage access**
2. Configure team/user permissions:
   - **Admin**: Repository maintainers
   - **Write**: Core contributors
   - **Read**: Regular contributors and external reviewers

## Step 7: Set Up Environments

Create deployment environments for better control:

1. Navigate to **Settings > Environments**
2. Create environments: `staging` and `production`

### Staging Environment Configuration
- **Deployment branches**: Selected branches only → `main`, `develop`
- **Environment secrets**: Add staging-specific secrets
- **Required reviewers**: Optional (for staging)
- **Wait timer**: 0 minutes

### Production Environment Configuration  
- **Deployment branches**: Selected branches only → `main`
- **Environment secrets**: Add production-specific secrets
- **Required reviewers**: Add 1-2 required reviewers
- **Wait timer**: 5 minutes (cooling-off period)

## Step 8: Configure Notifications

### GitHub Apps and Integrations

Consider installing these GitHub Apps for enhanced functionality:

1. **Codecov** - Code coverage reporting
   - Install from GitHub Marketplace
   - Authorize repository access
   - Configure coverage thresholds

2. **Dependabot** - Automated dependency updates
   - Usually enabled by default
   - Configure `.github/dependabot.yml` if needed

3. **SonarCloud** - Code quality analysis
   - Sign up at sonarcloud.io
   - Import repository
   - Configure quality gates

### Slack Integration (Optional)

1. Create Slack app or use existing webhook
2. Add webhook URL to repository secrets as `SLACK_WEBHOOK_URL`
3. Test notification by triggering a workflow

### Email Notifications

Configure notification preferences:
1. Go to **Settings > Notifications** (in your personal GitHub settings)
2. Configure **Actions** notifications:
   - ✅ Workflow runs
   - ✅ Failed workflows only (recommended)

## Step 9: Create Supporting Files

### Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.yml`:

```yaml
name: Bug Report
description: File a bug report
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: Version
      description: What version of the software are you running?
      placeholder: ex. v1.0.0
    validations:
      required: true
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: dropdown
    id: priority
    attributes:
      label: Priority
      description: How urgent is this issue?
      options:
        - Low
        - Medium
        - High
        - Critical
    validations:
      required: true
```

### Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global owners
* @senior-developer @tech-lead

# Source code
/src/ @senior-developer @python-expert

# Tests
/tests/ @qa-engineer @senior-developer

# Documentation
/docs/ @tech-writer @product-manager
*.md @tech-writer

# Configuration
/config/ @devops-engineer @tech-lead
/.github/ @devops-engineer @tech-lead
/docker-compose.yml @devops-engineer
/Dockerfile @devops-engineer

# Python specific
/pyproject.toml @python-expert @senior-developer
/poetry.lock @python-expert @senior-developer
```

## Step 10: Verify Workflow Setup

### Test Workflow Execution

1. Create a test branch: `git checkout -b test/workflow-validation`
2. Make a small change (e.g., update README.md)
3. Push branch and create a pull request
4. Verify that all workflows trigger correctly

### Validate Required Status Checks

1. Check that PR cannot be merged until all required checks pass
2. Verify that failing checks block merge
3. Test branch protection rules

### Monitor Workflow Performance

1. Navigate to **Actions** tab
2. Review workflow run times and success rates
3. Optimize slow workflows if needed

## Step 11: Documentation and Training

### Team Training

1. Document workflow processes for your team
2. Create troubleshooting guides for common issues
3. Set up training sessions on new CI/CD processes

### Update Project Documentation

1. Update README.md with build status badges:
   ```markdown
   ![CI](https://github.com/your-org/llm-cost-tracker/workflows/CI/badge.svg)
   ![Security](https://github.com/your-org/llm-cost-tracker/workflows/Security%20Scan/badge.svg)
   ```

2. Add contribution guidelines referencing the new workflows

## Troubleshooting Common Issues

### Workflow Not Triggering

**Problem**: Workflows don't run on push/PR
**Solutions**:
- Check workflow file syntax with `yamllint`
- Verify branch names match trigger patterns
- Ensure workflows are in correct directory (`.github/workflows/`)
- Check repository permissions

### Status Checks Not Required

**Problem**: PRs can be merged without status checks
**Solutions**:
- Verify status check names match job names in workflows
- Ensure branch protection rules are saved correctly
- Check that required status checks are enabled

### Secrets Not Working

**Problem**: Workflows fail with authentication errors
**Solutions**:
- Verify secret names match exactly (case-sensitive)
- Check secret values don't have trailing spaces
- Ensure secrets are available to the workflow scope

### Performance Issues

**Problem**: Workflows take too long to complete
**Solutions**:
- Implement dependency caching
- Use matrix builds for parallel execution
- Optimize Docker builds with multi-stage builds
- Review and remove unnecessary steps

## Maintenance Schedule

### Weekly
- Review failed workflow runs
- Update dependencies if automated updates are disabled
- Check for new security alerts

### Monthly  
- Review workflow performance metrics
- Update action versions to latest stable releases
- Clean up old workflow runs (if storage is a concern)

### Quarterly
- Audit repository access and permissions
- Review and update branch protection rules
- Assess workflow efficiency and optimization opportunities

## Support and Resources

### GitHub Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

### Community Resources
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)

### Getting Help
- GitHub Community Forum: [community.github.com](https://github.community)
- GitHub Support (for paid plans)
- Stack Overflow: Use `github-actions` tag

This completes the manual setup process for GitHub Actions workflows. The workflows should now be fully functional and provide comprehensive CI/CD capabilities for your LLM Cost Tracker project.