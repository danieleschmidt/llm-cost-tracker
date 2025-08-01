#!/usr/bin/env python3
"""
SDLC Implementation Validation Script

This script validates that all SDLC checkpoints have been properly implemented
and provides a health report of the repository's readiness for production use.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

class SDLCValidator:
    """Validates SDLC implementation completeness and quality."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.validation_results = {}
        self.issues_found = []
        self.recommendations = []
        
    def run_command(self, command: str, cwd: str = ".") -> Tuple[int, str, str]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def validate_checkpoint_1_foundation(self) -> Dict[str, Any]:
        """Validate Project Foundation & Documentation checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check core documentation files
        core_docs = [
            "README.md", "CHANGELOG.md", "LICENSE", "CONTRIBUTING.md", 
            "CODE_OF_CONDUCT.md", "SECURITY.md", "PROJECT_CHARTER.md"
        ]
        
        for doc in core_docs:
            if (self.repo_root / doc).exists():
                results["score"] += 10
                results["details"].append(f"✅ {doc} exists")
            else:
                results["details"].append(f"❌ {doc} missing")
                self.issues_found.append(f"Missing core documentation: {doc}")
        
        # Check docs structure
        docs_structure = [
            "docs/guides/README.md",
            "docs/adr/README.md", 
            "docs/adr/template.md",
            "docs/runbooks/README.md"
        ]
        
        for doc_path in docs_structure:
            if (self.repo_root / doc_path).exists():
                results["score"] += 5
                results["details"].append(f"✅ {doc_path} exists")
            else:
                results["details"].append(f"❌ {doc_path} missing")
        
        # Check ARCHITECTURE.md content quality
        arch_file = self.repo_root / "ARCHITECTURE.md"
        if arch_file.exists():
            content = arch_file.read_text()
            if len(content) > 1000 and "Components" in content:
                results["score"] += 10
                results["details"].append("✅ ARCHITECTURE.md is comprehensive")
            else:
                results["details"].append("⚠️ ARCHITECTURE.md needs enhancement")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_2_devenv(self) -> Dict[str, Any]:
        """Validate Development Environment & Tooling checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check configuration files
        config_files = [
            "pyproject.toml", ".editorconfig", ".gitignore", 
            ".pre-commit-config.yaml", "Makefile"
        ]
        
        for config in config_files:
            if (self.repo_root / config).exists():
                results["score"] += 15
                results["details"].append(f"✅ {config} exists")
            else:
                results["details"].append(f"❌ {config} missing")
                self.issues_found.append(f"Missing configuration: {config}")
        
        # Check VS Code configuration
        vscode_settings = self.repo_root / ".vscode/settings.json"
        if vscode_settings.exists():
            results["score"] += 10
            results["details"].append("✅ VS Code settings configured")
        else:
            results["details"].append("❌ VS Code settings missing")
        
        # Check devcontainer
        devcontainer = self.repo_root / ".devcontainer/devcontainer.json"
        if devcontainer.exists():
            results["score"] += 15
            results["details"].append("✅ DevContainer configured")
        else:
            results["details"].append("❌ DevContainer missing")
        
        # Validate pyproject.toml content
        pyproject = self.repo_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import toml
                config = toml.load(pyproject)
                if "tool.poetry.scripts" in config:
                    results["details"].append("✅ Poetry scripts configured")
                else:
                    results["details"].append("⚠️ Poetry scripts could be enhanced")
            except ImportError:
                results["details"].append("⚠️ Cannot validate pyproject.toml (toml package needed)")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_3_testing(self) -> Dict[str, Any]:
        """Validate Testing Infrastructure checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check test structure
        test_dirs = [
            "tests/unit", "tests/integration", "tests/e2e", 
            "tests/performance", "tests/fixtures"
        ]
        
        for test_dir in test_dirs:
            if (self.repo_root / test_dir).exists():
                results["score"] += 15
                results["details"].append(f"✅ {test_dir} exists")
            else:
                results["details"].append(f"❌ {test_dir} missing")
        
        # Check test configuration
        test_configs = ["pytest.ini", "tests/conftest.py"]
        for config in test_configs:
            if (self.repo_root / config).exists():
                results["score"] += 10
                results["details"].append(f"✅ {config} exists")
            else:
                results["details"].append(f"❌ {config} missing")
        
        # Check fixtures
        fixtures_files = [
            "tests/fixtures/sample_data.py",
            "tests/fixtures/mocks.py"
        ]
        for fixture in fixtures_files:
            if (self.repo_root / fixture).exists():
                results["score"] += 5
                results["details"].append(f"✅ {fixture} exists")
            else:
                results["details"].append(f"❌ {fixture} missing")
        
        # Try to run tests
        returncode, stdout, stderr = self.run_command("python -m pytest --version")
        if returncode == 0:
            results["score"] += 10
            results["details"].append("✅ pytest is available")
        else:
            results["details"].append("❌ pytest not available")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_4_build(self) -> Dict[str, Any]:
        """Validate Build & Containerization checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check build files
        build_files = [
            "Dockerfile", "docker-compose.yml", ".dockerignore"
        ]
        
        for build_file in build_files:
            if (self.repo_root / build_file).exists():
                results["score"] += 25
                results["details"].append(f"✅ {build_file} exists")
            else:
                results["details"].append(f"❌ {build_file} missing")
                self.issues_found.append(f"Missing build file: {build_file}")
        
        # Check Dockerfile quality
        dockerfile = self.repo_root / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text()
            if "FROM" in content and "COPY" in content:
                results["score"] += 10
                results["details"].append("✅ Dockerfile has basic structure")
                
                if "multi-stage" in content.lower() or len([l for l in content.split('\n') if l.strip().startswith('FROM')]) > 1:
                    results["score"] += 15
                    results["details"].append("✅ Multi-stage Dockerfile detected")
                else:
                    results["details"].append("⚠️ Consider multi-stage Dockerfile")
        
        # Check deployment documentation
        deploy_docs = [
            "docs/guides/developer-deployment.md"
        ]
        for doc in deploy_docs:
            if (self.repo_root / doc).exists():
                results["details"].append(f"✅ {doc} exists")
            else:
                results["details"].append(f"❌ {doc} missing")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_5_monitoring(self) -> Dict[str, Any]:
        """Validate Monitoring & Observability Setup checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check monitoring configuration
        monitoring_configs = [
            "config/prometheus.yml", "config/alert-rules.yml",
            "config/alertmanager.yml", "config/grafana-datasources.yml"
        ]
        
        for config in monitoring_configs:
            if (self.repo_root / config).exists():
                results["score"] += 20
                results["details"].append(f"✅ {config} exists")
            else:
                results["details"].append(f"❌ {config} missing")
        
        # Check runbooks
        runbooks = [
            "docs/runbooks/README.md",
            "docs/runbooks/high-cost-alerts.md"
        ]
        for runbook in runbooks:
            if (self.repo_root / runbook).exists():
                results["score"] += 10
                results["details"].append(f"✅ {runbook} exists")
            else:
                results["details"].append(f"❌ {runbook} missing")
        
        # Check monitoring guide
        monitoring_guide = self.repo_root / "docs/guides/admin-monitoring.md"
        if monitoring_guide.exists():
            results["score"] += 20
            results["details"].append("✅ Admin monitoring guide exists")
        else:
            results["details"].append("❌ Admin monitoring guide missing")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_6_workflows(self) -> Dict[str, Any]:
        """Validate Workflow Documentation & Templates checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check workflow documentation
        workflow_docs = [
            "docs/workflows/README.md",
            "docs/workflows/examples/README.md",
            "docs/workflows/MANUAL_SETUP_INSTRUCTIONS.md"
        ]
        
        for doc in workflow_docs:
            if (self.repo_root / doc).exists():
                results["score"] += 20
                results["details"].append(f"✅ {doc} exists")
            else:
                results["details"].append(f"❌ {doc} missing")
        
        # Check workflow examples
        workflow_examples = [
            "docs/workflows/examples/pr-validation.yml",
            "workflow-configs-ready-to-deploy/ci.yml",
            "workflow-configs-ready-to-deploy/security-scan.yml"
        ]
        
        for example in workflow_examples:
            if (self.repo_root / example).exists():
                results["score"] += 10
                results["details"].append(f"✅ {example} exists")
            else:
                results["details"].append(f"❌ {example} missing")
        
        # Check if workflows are actually deployed
        github_workflows = self.repo_root / ".github/workflows"
        if github_workflows.exists() and list(github_workflows.glob("*.yml")):
            results["score"] += 20
            results["details"].append("✅ GitHub Actions workflows deployed")
        else:
            results["details"].append("⚠️ GitHub Actions workflows need manual deployment")
            self.recommendations.append("Deploy workflow templates from docs/workflows/examples/")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_7_metrics(self) -> Dict[str, Any]:
        """Validate Metrics & Automation Setup checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check metrics configuration
        metrics_config = self.repo_root / ".github/project-metrics.json"
        if metrics_config.exists():
            results["score"] += 30
            results["details"].append("✅ Project metrics configuration exists")
            
            try:
                with open(metrics_config) as f:
                    config = json.load(f)
                    if "metrics" in config and "automation" in config:
                        results["score"] += 20
                        results["details"].append("✅ Metrics configuration is comprehensive")
                    else:
                        results["details"].append("⚠️ Metrics configuration needs enhancement")
            except json.JSONDecodeError:
                results["details"].append("❌ Metrics configuration has invalid JSON")
        else:
            results["details"].append("❌ Project metrics configuration missing")
        
        # Check automation scripts
        automation_scripts = [
            "scripts/metrics-collector.py",
            "scripts/repository-automation.py"
        ]
        
        for script in automation_scripts:
            if (self.repo_root / script).exists():
                results["score"] += 20
                results["details"].append(f"✅ {script} exists")
                
                # Check if script is executable
                returncode, stdout, stderr = self.run_command(f"python {script} --help")
                if returncode == 0:
                    results["score"] += 5
                    results["details"].append(f"✅ {script} is functional")
                else:
                    results["details"].append(f"⚠️ {script} may have issues")
            else:
                results["details"].append(f"❌ {script} missing")
        
        # Check automation guide
        automation_guide = self.repo_root / "docs/guides/admin-automation.md"
        if automation_guide.exists():
            results["score"] += 5
            results["details"].append("✅ Automation guide exists")
        else:
            results["details"].append("❌ Automation guide missing")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def validate_checkpoint_8_integration(self) -> Dict[str, Any]:
        """Validate Integration & Final Configuration checkpoint."""
        results = {"status": "pass", "score": 0, "max_score": 100, "details": []}
        
        # Check final documentation
        final_docs = [
            "SDLC_IMPLEMENTATION_COMPLETE.md"
        ]
        
        for doc in final_docs:
            if (self.repo_root / doc).exists():
                results["score"] += 50
                results["details"].append(f"✅ {doc} exists")
            else:
                results["details"].append(f"❌ {doc} missing")
        
        # Check overall repository health
        essential_files = [
            "README.md", "pyproject.toml", "Dockerfile", 
            "docs/guides/README.md", "tests/conftest.py"
        ]
        
        missing_files = [f for f in essential_files if not (self.repo_root / f).exists()]
        if not missing_files:
            results["score"] += 25
            results["details"].append("✅ All essential files present")
        else:
            results["details"].append(f"❌ Missing essential files: {missing_files}")
        
        # Check git repository health
        returncode, stdout, stderr = self.run_command("git status --porcelain")
        if returncode == 0:
            results["score"] += 25
            if stdout.strip():
                results["details"].append("ℹ️ Repository has uncommitted changes")
            else:
                results["details"].append("✅ Repository is clean")
        else:
            results["details"].append("❌ Git repository issues detected")
        
        if results["score"] < 80:
            results["status"] = "warning"
        if results["score"] < 60:
            results["status"] = "fail"
            
        return results
    
    def calculate_overall_score(self) -> Dict[str, Any]:
        """Calculate overall SDLC implementation score."""
        total_score = 0
        max_total_score = 0
        checkpoint_scores = []
        
        for checkpoint, results in self.validation_results.items():
            score = results["score"]
            max_score = results["max_score"]
            total_score += score
            max_total_score += max_score
            checkpoint_scores.append((checkpoint, score, max_score))
        
        overall_percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0
        
        if overall_percentage >= 90:
            status = "excellent"
            grade = "A"
        elif overall_percentage >= 80:
            status = "good"
            grade = "B"
        elif overall_percentage >= 70:
            status = "satisfactory"
            grade = "C"
        elif overall_percentage >= 60:
            status = "needs_improvement"
            grade = "D" 
        else:
            status = "poor"
            grade = "F"
        
        return {
            "overall_score": total_score,
            "max_score": max_total_score,
            "percentage": overall_percentage,
            "status": status,
            "grade": grade,
            "checkpoint_scores": checkpoint_scores
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = list(self.recommendations)  # Copy existing recommendations
        
        overall_results = self.calculate_overall_score()
        
        if overall_results["percentage"] < 90:
            recommendations.append("Consider addressing identified issues to achieve excellent status")
        
        if len(self.issues_found) > 0:
            recommendations.append(f"Resolve {len(self.issues_found)} critical issues found during validation")
        
        # Specific recommendations based on checkpoint results
        for checkpoint, results in self.validation_results.items():
            if results["status"] == "fail":
                recommendations.append(f"Priority: Fix critical issues in {checkpoint}")
            elif results["status"] == "warning":
                recommendations.append(f"Consider improvements to {checkpoint}")
        
        return recommendations
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete SDLC validation."""
        console.print("[blue]🔍 Starting SDLC Implementation Validation...[/blue]")
        
        checkpoints = [
            ("Checkpoint 1: Foundation", self.validate_checkpoint_1_foundation),
            ("Checkpoint 2: Dev Environment", self.validate_checkpoint_2_devenv),
            ("Checkpoint 3: Testing", self.validate_checkpoint_3_testing),
            ("Checkpoint 4: Build", self.validate_checkpoint_4_build),
            ("Checkpoint 5: Monitoring", self.validate_checkpoint_5_monitoring),
            ("Checkpoint 6: Workflows", self.validate_checkpoint_6_workflows),
            ("Checkpoint 7: Metrics", self.validate_checkpoint_7_metrics),
            ("Checkpoint 8: Integration", self.validate_checkpoint_8_integration),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for checkpoint_name, validator_func in checkpoints:
                task = progress.add_task(f"Validating {checkpoint_name}...", total=1)
                
                try:
                    results = validator_func()
                    self.validation_results[checkpoint_name] = results
                    progress.update(task, completed=1)
                except Exception as e:
                    console.print(f"[red]Error validating {checkpoint_name}: {e}[/red]")
                    self.validation_results[checkpoint_name] = {
                        "status": "error",
                        "score": 0,
                        "max_score": 100,
                        "details": [f"Validation error: {e}"]
                    }
                    progress.update(task, completed=1)
        
        overall_results = self.calculate_overall_score()
        recommendations = self.generate_recommendations()
        
        return {
            "overall": overall_results,
            "checkpoints": self.validation_results,
            "issues": self.issues_found,
            "recommendations": recommendations
        }

@click.command()
@click.option("--output", default="sdlc-validation-report.json", help="Output file for validation report")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--check-only", is_flag=True, help="Only check, don't generate report")
def main(output: str, verbose: bool, check_only: bool):
    """Validate SDLC implementation completeness."""
    
    validator = SDLCValidator()
    results = validator.run_validation()
    
    # Display results table
    console.print("\n[bold]📊 SDLC Implementation Validation Results[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Checkpoint", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Grade", justify="center")
    
    for checkpoint_name, checkpoint_results in results["checkpoints"].items():
        score = checkpoint_results["score"]
        max_score = checkpoint_results["max_score"]
        status = checkpoint_results["status"]
        
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        status_emoji = {
            "pass": "✅",
            "warning": "⚠️", 
            "fail": "❌",
            "error": "💥"
        }.get(status, "❓")
        
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        table.add_row(
            checkpoint_name.replace("Checkpoint ", ""),
            f"{score}/{max_score} ({percentage:.1f}%)",
            f"{status_emoji} {status.title()}",
            grade
        )
    
    console.print(table)
    
    # Overall score
    overall = results["overall"]
    
    console.print(f"\n[bold]🎯 Overall Score: {overall['overall_score']}/{overall['max_score']} ({overall['percentage']:.1f}%) - Grade {overall['grade']}[/bold]")
    
    status_colors = {
        "excellent": "green",
        "good": "blue", 
        "satisfactory": "yellow",
        "needs_improvement": "orange",
        "poor": "red"
    }
    status_color = status_colors.get(overall["status"], "white")
    console.print(f"[{status_color}]Status: {overall['status'].replace('_', ' ').title()}[/{status_color}]")
    
    # Issues found
    if results["issues"]:
        console.print(f"\n[red]⚠️ Issues Found ({len(results['issues'])}):[/red]")
        for issue in results["issues"][:10]:  # Show first 10 issues
            console.print(f"  • {issue}")
        if len(results["issues"]) > 10:
            console.print(f"  ... and {len(results['issues']) - 10} more issues")
    
    # Recommendations
    if results["recommendations"]:
        console.print(f"\n[blue]💡 Recommendations ({len(results['recommendations'])}):[/blue]")
        for rec in results["recommendations"]:
            console.print(f"  • {rec}")
    
    # Detailed results if verbose
    if verbose:
        console.print("\n[bold]📋 Detailed Results[/bold]")
        for checkpoint_name, checkpoint_results in results["checkpoints"].items():
            console.print(f"\n[cyan]{checkpoint_name}[/cyan]")
            for detail in checkpoint_results["details"]:
                console.print(f"  {detail}")
    
    # Save report
    if not check_only:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]📄 Detailed report saved to {output}[/green]")
    
    # Exit with appropriate code
    if overall["percentage"] < 70:
        console.print("\n[red]❌ SDLC implementation needs significant improvement[/red]")
        sys.exit(1)
    elif overall["percentage"] < 90:
        console.print("\n[yellow]⚠️ SDLC implementation is good but could be improved[/yellow]")
        sys.exit(0)
    else:
        console.print("\n[green]🎉 SDLC implementation is excellent![/green]")
        sys.exit(0)

if __name__ == "__main__":
    main()