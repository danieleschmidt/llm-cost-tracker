# Workflow Requirements Overview

This document outlines the GitHub Actions workflow requirements for this repository.

## Required Manual Setup (Admin Access Needed)

The following workflows require manual creation due to permission limitations:

### 1. CI/CD Pipeline
- **File**: `.github/workflows/ci.yml`
- **Purpose**: Automated testing, linting, security scans
- **Triggers**: Pull requests, pushes to main
- **Required Secrets**: None (uses public actions)
- **Dependencies**: Poetry, Python 3.11+

### 2. Security Scanning
- **File**: `.github/workflows/security.yml`
- **Purpose**: Dependency vulnerability scanning, secret detection
- **Triggers**: Daily, pull requests
- **Required Secrets**: None
- **Dependencies**: Bandit, Safety, TruffleHog

### 3. Release Automation
- **File**: `.github/workflows/release.yml`
- **Purpose**: Automated releases with semantic versioning
- **Triggers**: Tags matching v*
- **Required Secrets**: PyPI_TOKEN (for package publishing)
- **Dependencies**: Poetry, GitHub CLI

### 4. Documentation
- **File**: `.github/workflows/docs.yml`
- **Purpose**: Build and deploy documentation
- **Triggers**: Pushes to main
- **Required Secrets**: None
- **Dependencies**: Sphinx, GitHub Pages

## Branch Protection Requirements

Configure the following branch protection rules for `main`:
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to admins only

## Repository Settings

Required manual configuration:
- Topics: `python`, `opentelemetry`, `cost-tracking`, `llm`
- Description: "Self-hostable OpenTelemetry collector for LLM cost tracking"
- Homepage: Link to documentation site
- Enable security advisories
- Enable Dependabot alerts

For detailed setup instructions, see [docs/SETUP_REQUIRED.md](../SETUP_REQUIRED.md).