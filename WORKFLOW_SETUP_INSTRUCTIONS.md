# 🔧 Workflow Setup Instructions

## Important: Manual Workflow Setup Required

Due to GitHub security restrictions, workflow files cannot be automatically created by GitHub Apps without the `workflows` permission. The SBOM generation workflow has been provided as a template that requires manual setup.

## 📋 Setup Steps

### 1. Copy Workflow Template

Copy the workflow template to activate it:

```bash
# From the repository root
cp .github/workflow-templates/sbom-generation.yml .github/workflows/sbom-generation.yml
```

### 2. Commit the Workflow

```bash
git add .github/workflows/sbom-generation.yml
git commit -m "feat: add SBOM generation and supply chain security workflow"
git push
```

### 3. Configure Required Secrets (Optional)

For enhanced functionality, add these repository secrets:

- `COSIGN_PRIVATE_KEY`: For artifact signing (optional)
- `SLACK_WEBHOOK_URL`: For notifications (optional)

### 4. Verify Workflow

1. Go to the Actions tab in your GitHub repository
2. You should see the "🔒 SBOM Generation and Supply Chain Security" workflow
3. The workflow will run automatically on pushes to main and can be triggered manually

## 🔍 What This Workflow Provides

- **Automated SBOM Generation**: Creates comprehensive software bills of materials
- **Vulnerability Scanning**: Scans for security vulnerabilities in dependencies
- **Supply Chain Security**: Generates SLSA provenance attestations
- **Multiple Formats**: Supports SPDX and CycloneDX SBOM formats
- **Container Scanning**: Includes Docker image security analysis
- **Compliance Reporting**: Generates security compliance reports

## 🚀 Benefits

- **Transparency**: Complete visibility into software supply chain
- **Security**: Automated vulnerability detection and reporting
- **Compliance**: Meets SLSA, SPDX, and CycloneDX standards
- **Integration**: Ready for enterprise security tools and policies

## 💡 Alternative: Manual SBOM Generation

If you prefer not to use the automated workflow, you can generate SBOMs manually using the provided script:

```bash
# Generate SBOM using the script
./scripts/generate-sbom.sh

# View generated SBOM files
ls -la sbom/
```

## 📞 Support

If you encounter issues with the workflow setup, please:

1. Check that the workflow file was copied correctly
2. Verify your repository has Actions enabled
3. Ensure you have the necessary permissions
4. Review the workflow logs for any error messages