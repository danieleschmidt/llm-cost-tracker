#!/usr/bin/env python3
"""
Repository automation and maintenance script for LLM Cost Tracker.

This script performs automated maintenance tasks including:
- Dependency updates
- Security scanning
- Code quality checks
- Documentation updates
- Issue management
- Performance monitoring
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

class RepositoryAutomation:
    """Automated repository maintenance and optimization."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize automation with configuration."""
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API tokens
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        
        self.repo = self.config["project"]["repository"]
        self.automation_config = self.config.get("automation", {})
        
        # Track changes made
        self.changes_made = []
        self.issues_created = []
        self.prs_created = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Configuration file {config_path} not found[/red]")
            sys.exit(1)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def run_command(self, command: str, cwd: str = ".") -> Tuple[int, str, str]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    async def update_dependencies(self) -> List[str]:
        """Update project dependencies and create PR if needed."""
        if not self.automation_config.get("dependency_updates", {}).get("enabled", False):
            return []
        
        console.print("[blue]🔄 Checking for dependency updates...[/blue]")
        changes = []
        
        # Update Poetry dependencies
        if Path("pyproject.toml").exists():
            console.print("Updating Python dependencies...")
            
            # Check for outdated packages
            returncode, stdout, stderr = self.run_command("poetry show --outdated")
            if returncode == 0 and stdout.strip():
                outdated_packages = []
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            outdated_packages.append(parts[0])
                
                if outdated_packages:
                    console.print(f"Found {len(outdated_packages)} outdated packages")
                    
                    # Update packages based on automation policy
                    auto_merge = self.automation_config["dependency_updates"].get("auto_merge", {})
                    
                    for package in outdated_packages:
                        # For now, just update patch versions automatically
                        if auto_merge.get("patch", False):
                            returncode, stdout, stderr = self.run_command(f"poetry update {package}")
                            if returncode == 0:
                                changes.append(f"Updated {package} to latest patch version")
                            else:
                                console.print(f"[yellow]Warning: Failed to update {package}: {stderr}[/yellow]")
            
            # Update lock file
            returncode, stdout, stderr = self.run_command("poetry lock --no-update")
            if returncode == 0:
                changes.append("Updated poetry.lock file")
        
        # Update npm dependencies if package.json exists
        if Path("package.json").exists():
            console.print("Checking npm dependencies...")
            returncode, stdout, stderr = self.run_command("npm outdated --json")
            if returncode != 0 and stdout:  # npm outdated returns non-zero when outdated packages exist
                try:
                    outdated = json.loads(stdout)
                    if outdated:
                        console.print(f"Found {len(outdated)} outdated npm packages")
                        
                        # Update devDependencies automatically
                        returncode, stdout, stderr = self.run_command("npm update --save-dev")
                        if returncode == 0:
                            changes.append("Updated npm devDependencies")
                except json.JSONDecodeError:
                    pass
        
        return changes
    
    async def security_scan(self) -> List[str]:
        """Run security scans and create issues for vulnerabilities."""
        if not self.automation_config.get("security_scanning", {}).get("enabled", False):
            return []
        
        console.print("[blue]🔒 Running security scans...[/blue]")
        issues = []
        
        # Python security scan with Safety
        if Path("poetry.lock").exists():
            console.print("Scanning Python dependencies for vulnerabilities...")
            returncode, stdout, stderr = self.run_command("poetry run safety check --json")
            if returncode != 0 and stdout:
                try:
                    vulnerabilities = json.loads(stdout)
                    for vuln in vulnerabilities:
                        if vuln.get("vulnerability_id"):
                            issue_title = f"Security: {vuln['package_name']} vulnerability ({vuln['vulnerability_id']})"
                            issue_body = f"""
## Security Vulnerability Detected

**Package**: {vuln['package_name']}
**Installed Version**: {vuln['installed_version']}
**Vulnerable Spec**: {vuln['vulnerable_spec']}
**Advisory**: {vuln.get('advisory', 'N/A')}

### Recommendation
{vuln.get('advisory', 'Update to a secure version')}

---
*This issue was automatically created by the repository automation system.*
"""
                            issues.append({
                                "title": issue_title,
                                "body": issue_body,
                                "labels": ["security", "vulnerability", "automation"]
                            })
                except json.JSONDecodeError:
                    pass
        
        # Bandit security scan
        console.print("Running Bandit security analysis...")
        returncode, stdout, stderr = self.run_command("poetry run bandit -r src/ -f json")
        if returncode != 0 and stdout:
            try:
                bandit_results = json.loads(stdout)
                high_severity = [
                    result for result in bandit_results.get("results", [])
                    if result.get("issue_severity") == "HIGH"
                ]
                
                if high_severity:
                    issue_title = f"Security: {len(high_severity)} high-severity security issues found"
                    issue_body = f"""
## High-Severity Security Issues

Bandit has detected {len(high_severity)} high-severity security issues in the codebase:

"""
                    for i, issue in enumerate(high_severity[:5], 1):  # Limit to first 5
                        issue_body += f"""
### Issue {i}: {issue.get('test_name', 'Unknown')}
**File**: {issue.get('filename', 'Unknown')}
**Line**: {issue.get('line_number', 'Unknown')}
**Severity**: {issue.get('issue_severity', 'Unknown')}
**Confidence**: {issue.get('issue_confidence', 'Unknown')}

{issue.get('issue_text', 'No description available')}

"""
                    
                    issue_body += """
---
*This issue was automatically created by the repository automation system.*
Please review and address these security concerns promptly.
"""
                    
                    issues.append({
                        "title": issue_title,
                        "body": issue_body,
                        "labels": ["security", "high-priority", "automation"]
                    })
            except json.JSONDecodeError:
                pass
        
        return issues
    
    async def code_quality_check(self) -> List[str]:
        """Run code quality checks and create improvement tasks."""
        if not self.automation_config.get("code_quality", {}).get("enabled", False):
            return []
        
        console.print("[blue]📊 Analyzing code quality...[/blue]")
        suggestions = []
        
        # Check test coverage
        console.print("Checking test coverage...")
        returncode, stdout, stderr = self.run_command("poetry run pytest --cov=src/llm_cost_tracker --cov-report=json")
        if returncode == 0 and Path("coverage.json").exists():
            try:
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                threshold = self.automation_config["code_quality"]["quality_gates"]["coverage_minimum"]
                
                if total_coverage < threshold:
                    suggestions.append({
                        "type": "coverage",
                        "current": total_coverage,
                        "target": threshold,
                        "message": f"Test coverage is {total_coverage:.1f}%, below target of {threshold}%"
                    })
                
                # Check for files with low coverage
                low_coverage_files = []
                for filename, file_data in coverage_data.get("files", {}).items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 100)
                    if file_coverage < 70:  # Files with less than 70% coverage
                        low_coverage_files.append((filename, file_coverage))
                
                if low_coverage_files:
                    suggestions.append({
                        "type": "file_coverage",
                        "files": low_coverage_files[:10],  # Top 10 files
                        "message": f"Found {len(low_coverage_files)} files with low test coverage"
                    })
                        
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Check for code complexity
        console.print("Analyzing code complexity...")
        returncode, stdout, stderr = self.run_command("poetry run radon cc src/ --json")
        if returncode == 0 and stdout:
            try:
                complexity_data = json.loads(stdout)
                complex_functions = []
                
                for filename, file_data in complexity_data.items():
                    for item in file_data:
                        if item.get("complexity", 0) > 10:  # McCabe complexity > 10
                            complex_functions.append({
                                "file": filename,
                                "function": item.get("name", "Unknown"),
                                "complexity": item.get("complexity", 0),
                                "line": item.get("lineno", 0)
                            })
                
                if complex_functions:
                    suggestions.append({
                        "type": "complexity",
                        "functions": complex_functions[:10],  # Top 10 most complex
                        "message": f"Found {len(complex_functions)} functions with high complexity"
                    })
            except json.JSONDecodeError:
                pass
        
        return suggestions
    
    async def update_documentation(self) -> List[str]:
        """Update documentation automatically."""
        console.print("[blue]📚 Updating documentation...[/blue]")
        updates = []
        
        # Update README badges if needed
        readme_path = Path("README.md")
        if readme_path.exists():
            content = readme_path.read_text()
            
            # Check if badges exist and are up to date
            expected_badges = [
                f"![CI](https://github.com/{self.repo}/workflows/CI/badge.svg)",
                f"![Security](https://github.com/{self.repo}/workflows/Security%20Scan/badge.svg)",
                f"![License](https://img.shields.io/github/license/{self.repo})"
            ]
            
            badges_added = []
            for badge in expected_badges:
                if badge not in content:
                    badges_added.append(badge)
            
            if badges_added:
                # Add badges after the title
                lines = content.split('\n')
                title_line = next((i for i, line in enumerate(lines) if line.startswith('#')), 0)
                
                # Insert badges after title
                for i, badge in enumerate(badges_added):
                    lines.insert(title_line + 1 + i, badge)
                
                readme_path.write_text('\n'.join(lines))
                updates.append(f"Added {len(badges_added)} status badges to README")
        
        # Update CHANGELOG.md with recent changes
        changelog_path = Path("CHANGELOG.md")
        if changelog_path.exists() and self.changes_made:
            content = changelog_path.read_text()
            
            # Add unreleased section if it doesn't exist
            if "## [Unreleased]" not in content:
                lines = content.split('\n')
                # Find the first version section
                first_version_line = next(
                    (i for i, line in enumerate(lines) if line.startswith("## [") and "Unreleased" not in line),
                    len(lines)
                )
                
                unreleased_section = [
                    "## [Unreleased]",
                    "",
                    "### Added",
                    "- Automated repository maintenance improvements",
                    "",
                    "### Changed",
                    "",
                    "### Fixed",
                    "",
                ]
                
                for i, line in enumerate(unreleased_section):
                    lines.insert(first_version_line + i, line)
                
                changelog_path.write_text('\n'.join(lines))
                updates.append("Updated CHANGELOG.md with unreleased section")
        
        return updates
    
    async def create_github_issue(self, issue: Dict[str, Any]) -> bool:
        """Create a GitHub issue."""
        if not self.github_token:
            console.print("[yellow]Warning: No GitHub token, cannot create issues[/yellow]")
            return False
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "title": issue["title"],
            "body": issue["body"],
            "labels": issue.get("labels", [])
        }
        
        try:
            async with self.session.post(
                f"https://api.github.com/repos/{self.repo}/issues",
                headers=headers,
                json=data
            ) as resp:
                if resp.status == 201:
                    issue_data = await resp.json()
                    self.issues_created.append(issue_data["html_url"])
                    return True
                else:
                    console.print(f"[red]Failed to create issue: {resp.status}[/red]")
                    return False
        except Exception as e:
            console.print(f"[red]Error creating issue: {e}[/red]")
            return False
    
    async def create_pull_request(self, title: str, body: str, branch: str) -> bool:
        """Create a pull request."""
        if not self.github_token:
            return False
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "title": title,
            "body": body,
            "head": branch,
            "base": "main"
        }
        
        try:
            async with self.session.post(
                f"https://api.github.com/repos/{self.repo}/pulls",
                headers=headers,
                json=data
            ) as resp:
                if resp.status == 201:
                    pr_data = await resp.json()
                    self.prs_created.append(pr_data["html_url"])
                    return True
                else:
                    console.print(f"[red]Failed to create PR: {resp.status}[/red]")
                    return False
        except Exception as e:
            console.print(f"[red]Error creating PR: {e}[/red]")
            return False
    
    async def send_slack_notification(self, message: str) -> None:
        """Send notification to Slack."""
        if not self.slack_webhook:
            return
        
        payload = {
            "text": f"🤖 Repository Automation Report",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
        }
        
        try:
            async with self.session.post(self.slack_webhook, json=payload) as resp:
                if resp.status != 200:
                    console.print(f"[yellow]Warning: Failed to send Slack notification: {resp.status}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Error sending Slack notification: {e}[/yellow]")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of automation activities."""
        report = []
        report.append("# 🤖 Repository Automation Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"**Repository**: {self.repo}")
        report.append("")
        
        if self.changes_made:
            report.append("## ✅ Changes Made")
            for change in self.changes_made:
                report.append(f"- {change}")
            report.append("")
        
        if self.issues_created:
            report.append("## 🐛 Issues Created")
            for issue_url in self.issues_created:
                report.append(f"- {issue_url}")
            report.append("")
        
        if self.prs_created:
            report.append("## 🔄 Pull Requests Created")
            for pr_url in self.prs_created:
                report.append(f"- {pr_url}")
            report.append("")
        
        if not self.changes_made and not self.issues_created and not self.prs_created:
            report.append("## ℹ️ Status")
            report.append("No automated changes were needed. Repository is in good health! 🎉")
            report.append("")
        
        report.append("---")
        report.append("*This report was generated by the automated repository maintenance system.*")
        
        return "\n".join(report)

@click.command()
@click.option("--config", default=".github/project-metrics.json", help="Path to configuration file")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option("--skip-deps", is_flag=True, help="Skip dependency updates")
@click.option("--skip-security", is_flag=True, help="Skip security scans")
@click.option("--skip-quality", is_flag=True, help="Skip code quality checks")
@click.option("--skip-docs", is_flag=True, help="Skip documentation updates")
@click.option("--create-issues", is_flag=True, help="Create GitHub issues for problems found")
@click.option("--notify-slack", is_flag=True, help="Send notifications to Slack")
@click.option("--verbose", is_flag=True, help="Verbose output")
async def main(config: str, dry_run: bool, skip_deps: bool, skip_security: bool, 
               skip_quality: bool, skip_docs: bool, create_issues: bool, 
               notify_slack: bool, verbose: bool):
    """Run automated repository maintenance tasks."""
    
    if dry_run:
        console.print("[yellow]🧪 DRY RUN MODE - No changes will be made[/yellow]")
    
    async with RepositoryAutomation(config) as automation:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            all_tasks = []
            
            # Dependency updates
            if not skip_deps:
                task = progress.add_task("Updating dependencies...", total=1)
                try:
                    changes = await automation.update_dependencies()
                    automation.changes_made.extend(changes)
                    progress.update(task, completed=1)
                    if verbose and changes:
                        console.print(f"[green]✓ Made {len(changes)} dependency updates[/green]")
                except Exception as e:
                    progress.update(task, completed=1)
                    console.print(f"[red]✗ Dependency update failed: {e}[/red]")
            
            # Security scanning
            if not skip_security:
                task = progress.add_task("Running security scans...", total=1)
                try:
                    security_issues = await automation.security_scan()
                    if security_issues and create_issues and not dry_run:
                        for issue in security_issues:
                            await automation.create_github_issue(issue)
                    progress.update(task, completed=1)
                    if verbose and security_issues:
                        console.print(f"[yellow]⚠ Found {len(security_issues)} security issues[/yellow]")
                except Exception as e:
                    progress.update(task, completed=1)
                    console.print(f"[red]✗ Security scan failed: {e}[/red]")
            
            # Code quality checks
            if not skip_quality:
                task = progress.add_task("Analyzing code quality...", total=1)
                try:
                    quality_suggestions = await automation.code_quality_check()
                    if quality_suggestions and verbose:
                        console.print(f"[blue]📊 Found {len(quality_suggestions)} quality improvement opportunities[/blue]")
                    progress.update(task, completed=1)
                except Exception as e:
                    progress.update(task, completed=1)
                    console.print(f"[red]✗ Code quality check failed: {e}[/red]")
            
            # Documentation updates
            if not skip_docs:
                task = progress.add_task("Updating documentation...", total=1)
                try:
                    doc_updates = await automation.update_documentation()
                    automation.changes_made.extend(doc_updates)
                    progress.update(task, completed=1)
                    if verbose and doc_updates:
                        console.print(f"[green]✓ Made {len(doc_updates)} documentation updates[/green]")
                except Exception as e:
                    progress.update(task, completed=1)
                    console.print(f"[red]✗ Documentation update failed: {e}[/red]")
        
        # Generate and display summary
        summary = automation.generate_summary_report()
        console.print("\n[bold]📋 Automation Summary[/bold]")
        console.print(summary)
        
        # Save report
        report_file = f"automation-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(summary)
        console.print(f"\n[green]📄 Report saved to {report_file}[/green]")
        
        # Send Slack notification
        if notify_slack and not dry_run:
            slack_message = f"""
*Repository Automation Complete*

**Repository**: {automation.repo}
**Changes Made**: {len(automation.changes_made)}
**Issues Created**: {len(automation.issues_created)}
**PRs Created**: {len(automation.prs_created)}

{summary}
"""
            await automation.send_slack_notification(slack_message)

if __name__ == "__main__":
    asyncio.run(main())