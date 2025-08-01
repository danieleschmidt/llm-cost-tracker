#!/usr/bin/env python3
"""
Automated metrics collection and reporting system for LLM Cost Tracker.

This script collects metrics from various sources (GitHub, Prometheus, SonarQube, etc.)
and generates reports for project health monitoring.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncpg
import click
import yaml
from prometheus_client.parser import text_string_to_metric_families
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

class MetricsCollector:
    """Collect and aggregate metrics from multiple sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize metrics collector with configuration."""
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # API tokens and credentials
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.sonar_token = os.getenv("SONAR_TOKEN")
        self.codecov_token = os.getenv("CODECOV_TOKEN")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Configuration file {config_path} not found[/red]")
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in {config_path}: {e}[/red]")
            sys.exit(1)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        
        # Initialize database connection if configured
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                self.db_pool = await asyncpg.create_pool(db_url)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not connect to database: {e}[/yellow]")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()
    
    async def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.github_token:
            console.print("[yellow]Warning: No GitHub token provided, skipping GitHub metrics[/yellow]")
            return {}
        
        repo = self.config["project"]["repository"]
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        metrics = {}
        
        try:
            # Repository info
            async with self.session.get(f"https://api.github.com/repos/{repo}", headers=headers) as resp:
                if resp.status == 200:
                    repo_data = await resp.json()
                    metrics.update({
                        "stars": repo_data.get("stargazers_count", 0),
                        "forks": repo_data.get("forks_count", 0),
                        "open_issues": repo_data.get("open_issues_count", 0),
                        "watchers": repo_data.get("watchers_count", 0),
                        "size_kb": repo_data.get("size", 0)
                    })
            
            # Pull requests (last 30 days)
            since_date = (datetime.now() - timedelta(days=30)).isoformat()
            async with self.session.get(
                f"https://api.github.com/repos/{repo}/pulls",
                headers=headers,
                params={"state": "all", "since": since_date}
            ) as resp:
                if resp.status == 200:
                    prs = await resp.json()
                    merged_prs = [pr for pr in prs if pr.get("merged_at")]
                    metrics.update({
                        "pull_requests_total": len(prs),
                        "pull_requests_merged": len(merged_prs),
                        "pull_requests_open": len([pr for pr in prs if pr.get("state") == "open"])
                    })
            
            # Commits (last 30 days)
            async with self.session.get(
                f"https://api.github.com/repos/{repo}/commits",
                headers=headers,
                params={"since": since_date}
            ) as resp:
                if resp.status == 200:
                    commits = await resp.json()
                    unique_authors = set(commit["commit"]["author"]["name"] for commit in commits if commit.get("commit", {}).get("author", {}).get("name"))
                    metrics.update({
                        "commits_count": len(commits),
                        "active_contributors": len(unique_authors)
                    })
            
            # Security alerts
            async with self.session.get(
                f"https://api.github.com/repos/{repo}/vulnerability-alerts",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    alerts = await resp.json()
                    metrics["security_alerts"] = len(alerts)
                elif resp.status == 404:
                    metrics["security_alerts"] = 0
            
            # Workflows (Actions)
            async with self.session.get(
                f"https://api.github.com/repos/{repo}/actions/workflows",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    workflows = await resp.json()
                    total_runs = 0
                    successful_runs = 0
                    
                    for workflow in workflows.get("workflows", []):
                        workflow_id = workflow["id"]
                        async with self.session.get(
                            f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/runs",
                            headers=headers,
                            params={"per_page": 10}
                        ) as run_resp:
                            if run_resp.status == 200:
                                runs = await run_resp.json()
                                workflow_runs = runs.get("workflow_runs", [])
                                total_runs += len(workflow_runs)
                                successful_runs += len([r for r in workflow_runs if r.get("conclusion") == "success"])
                    
                    metrics.update({
                        "workflow_runs_total": total_runs,
                        "workflow_success_rate": successful_runs / max(total_runs, 1) * 100
                    })
        
        except Exception as e:
            console.print(f"[red]Error collecting GitHub metrics: {e}[/red]")
        
        return metrics
    
    async def collect_prometheus_metrics(self) -> Dict[str, Any]:
        """Collect metrics from Prometheus."""
        prometheus_config = self.config["tracking"]["data_sources"].get("prometheus", {})
        endpoint = prometheus_config.get("endpoint", "http://localhost:9090")
        
        if not endpoint:
            return {}
        
        metrics = {}
        
        try:
            # Application uptime
            query = "up{job='llm-cost-tracker'}"
            async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("data", {}).get("result", [])
                    if result:
                        metrics["application_uptime"] = float(result[0]["value"][1])
            
            # Request rate (last 5 minutes)
            query = "rate(llm_cost_tracker_requests_total[5m])"
            async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("data", {}).get("result", [])
                    if result:
                        metrics["request_rate_5m"] = sum(float(r["value"][1]) for r in result)
            
            # Error rate (last 5 minutes)
            query = "rate(llm_cost_tracker_errors_total[5m]) / rate(llm_cost_tracker_requests_total[5m]) * 100"
            async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("data", {}).get("result", [])
                    if result:
                        metrics["error_rate_percentage"] = float(result[0]["value"][1])
            
            # Response time percentiles
            for percentile in [50, 95, 99]:
                query = f"histogram_quantile(0.{percentile}, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))"
                async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("data", {}).get("result", [])
                        if result:
                            metrics[f"response_time_p{percentile}_ms"] = float(result[0]["value"][1]) * 1000
            
            # Cost metrics
            query = "llm_cost_tracker_total_cost_usd"
            async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("data", {}).get("result", [])
                    total_cost = sum(float(r["value"][1]) for r in result)
                    metrics["total_cost_usd"] = total_cost
            
            # Hourly cost rate
            query = "rate(llm_cost_tracker_total_cost_usd[1h])"
            async with self.session.get(f"{endpoint}/api/v1/query", params={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("data", {}).get("result", [])
                    if result:
                        metrics["hourly_cost_rate"] = sum(float(r["value"][1]) for r in result)
        
        except Exception as e:
            console.print(f"[red]Error collecting Prometheus metrics: {e}[/red]")
        
        return metrics
    
    async def collect_sonarqube_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics from SonarQube/SonarCloud."""
        if not self.sonar_token:
            console.print("[yellow]Warning: No SonarQube token provided, skipping code quality metrics[/yellow]")
            return {}
        
        sonar_config = self.config["tracking"]["data_sources"].get("sonarqube", {})
        endpoint = sonar_config.get("endpoint", "https://sonarcloud.io")
        project_key = sonar_config.get("project_key")
        
        if not project_key:
            return {}
        
        headers = {"Authorization": f"Bearer {self.sonar_token}"}
        metrics = {}
        
        try:
            # Project measures
            metric_keys = [
                "coverage", "duplicated_lines_density", "ncloc", "complexity",
                "code_smells", "bugs", "vulnerabilities", "security_hotspots",
                "sqale_rating", "reliability_rating", "security_rating",
                "sqale_index"
            ]
            
            params = {
                "component": project_key,
                "metricKeys": ",".join(metric_keys)
            }
            
            async with self.session.get(
                f"{endpoint}/api/measures/component",
                headers=headers,
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    measures = data.get("component", {}).get("measures", [])
                    
                    for measure in measures:
                        metric_key = measure["metric"]
                        value = measure.get("value")
                        if value is not None:
                            # Convert rating letters to numbers
                            if metric_key.endswith("_rating"):
                                rating_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
                                metrics[f"sonar_{metric_key}"] = rating_map.get(value, 5)
                            else:
                                try:
                                    metrics[f"sonar_{metric_key}"] = float(value)
                                except ValueError:
                                    metrics[f"sonar_{metric_key}"] = value
        
        except Exception as e:
            console.print(f"[red]Error collecting SonarQube metrics: {e}[/red]")
        
        return metrics
    
    async def collect_database_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics from database."""
        if not self.db_pool:
            return {}
        
        metrics = {}
        
        try:
            async with self.db_pool.acquire() as conn:
                # Cost tracking metrics
                cost_query = """
                    SELECT 
                        COUNT(*) as total_requests,
                        SUM(cost_usd) as total_cost,
                        AVG(cost_usd) as avg_cost_per_request,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT application) as unique_applications
                    FROM cost_tracking.cost_records 
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                """
                
                row = await conn.fetchrow(cost_query)
                if row:
                    metrics.update({
                        "daily_requests": row["total_requests"] or 0,
                        "daily_cost_usd": float(row["total_cost"] or 0),
                        "avg_cost_per_request": float(row["avg_cost_per_request"] or 0),
                        "daily_unique_users": row["unique_users"] or 0,
                        "active_applications": row["unique_applications"] or 0
                    })
                
                # Model usage breakdown
                model_query = """
                    SELECT 
                        model,
                        COUNT(*) as requests,
                        SUM(cost_usd) as cost,
                        AVG(total_tokens) as avg_tokens
                    FROM cost_tracking.cost_records 
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY model
                    ORDER BY cost DESC
                    LIMIT 10
                """
                
                model_rows = await conn.fetch(model_query)
                model_metrics = {}
                for row in model_rows:
                    model_name = row["model"].replace("-", "_").replace(".", "_")
                    model_metrics[f"model_{model_name}_requests"] = row["requests"]
                    model_metrics[f"model_{model_name}_cost"] = float(row["cost"])
                    model_metrics[f"model_{model_name}_avg_tokens"] = float(row["avg_tokens"] or 0)
                
                metrics.update(model_metrics)
                
                # Error tracking
                error_query = """
                    SELECT 
                        COUNT(*) as error_count
                    FROM application_logs 
                    WHERE level = 'ERROR' 
                    AND timestamp >= NOW() - INTERVAL '1 hour'
                """
                
                error_row = await conn.fetchrow(error_query)
                if error_row:
                    metrics["hourly_error_count"] = error_row["error_count"] or 0
        
        except Exception as e:
            console.print(f"[red]Error collecting database metrics: {e}[/red]")
        
        return metrics
    
    async def store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store collected metrics in database."""
        if not self.db_pool:
            # Store in file if no database
            timestamp = datetime.now().isoformat()
            metrics_file = f"metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "metrics": metrics
                }, f, indent=2)
            console.print(f"[green]Metrics stored in {metrics_file}[/green]")
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Create metrics table if it doesn't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        metric_name VARCHAR(255) NOT NULL,
                        metric_value NUMERIC,
                        metric_metadata JSONB,
                        source VARCHAR(100)
                    )
                """)
                
                # Insert metrics
                for metric_name, metric_value in metrics.items():
                    source = self._determine_metric_source(metric_name)
                    
                    if isinstance(metric_value, (int, float)):
                        await conn.execute("""
                            INSERT INTO project_metrics (metric_name, metric_value, source)
                            VALUES ($1, $2, $3)
                        """, metric_name, metric_value, source)
                    else:
                        await conn.execute("""
                            INSERT INTO project_metrics (metric_name, metric_metadata, source)
                            VALUES ($1, $2, $3)
                        """, metric_name, json.dumps({"value": metric_value}), source)
        
        except Exception as e:
            console.print(f"[red]Error storing metrics: {e}[/red]")
    
    def _determine_metric_source(self, metric_name: str) -> str:
        """Determine the source of a metric based on its name."""
        if metric_name.startswith("sonar_"):
            return "sonarqube"
        elif metric_name in ["stars", "forks", "commits_count", "pull_requests_total"]:
            return "github"
        elif metric_name in ["application_uptime", "request_rate_5m", "error_rate_percentage"]:
            return "prometheus"
        elif metric_name in ["daily_requests", "daily_cost_usd", "hourly_error_count"]:
            return "database"
        else:
            return "unknown"
    
    def generate_health_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall project health score based on collected metrics."""
        thresholds = self.config["metrics"]
        
        scores = {}
        
        # Code Quality Score (0-100)
        coverage = metrics.get("sonar_coverage", 0)
        code_quality_score = min(100, max(0, coverage))
        if "sonar_sqale_rating" in metrics:
            # Lower rating is better (A=1, B=2, etc.)
            rating_bonus = max(0, (6 - metrics["sonar_sqale_rating"]) * 10)
            code_quality_score = min(100, code_quality_score + rating_bonus)
        scores["code_quality"] = code_quality_score
        
        # Reliability Score (0-100)
        uptime = metrics.get("application_uptime", 0) * 100
        error_rate = metrics.get("error_rate_percentage", 100)
        reliability_score = min(100, max(0, uptime - error_rate))
        scores["reliability"] = reliability_score
        
        # Performance Score (0-100)
        response_time_p95 = metrics.get("response_time_p95_ms", 5000)
        performance_threshold = thresholds["performance"]["response_time_p95_threshold_ms"]
        performance_score = max(0, min(100, 100 - (response_time_p95 / performance_threshold * 100)))
        scores["performance"] = performance_score
        
        # Security Score (0-100)
        vulnerabilities = metrics.get("sonar_vulnerabilities", 0)
        security_hotspots = metrics.get("sonar_security_hotspots", 0)
        security_score = max(0, 100 - (vulnerabilities * 10 + security_hotspots * 5))
        scores["security"] = security_score
        
        # Overall Health Score (weighted average)
        overall_score = (
            scores["code_quality"] * 0.25 +
            scores["reliability"] * 0.30 +
            scores["performance"] * 0.25 +
            scores["security"] * 0.20
        )
        scores["overall"] = overall_score
        
        return scores
    
    def generate_report(self, metrics: Dict[str, Any], health_scores: Dict[str, Any]) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("# LLM Cost Tracker - Project Health Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        
        # Health Scores
        report.append("## 📊 Health Scores")
        report.append("")
        for category, score in health_scores.items():
            status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            report.append(f"- **{category.replace('_', ' ').title()}**: {score:.1f}/100 {status_emoji}")
        report.append("")
        
        # Key Metrics
        report.append("## 📈 Key Metrics")
        report.append("")
        
        # GitHub metrics
        if any(k in metrics for k in ["stars", "forks", "commits_count"]):
            report.append("### GitHub Repository")
            if "stars" in metrics:
                report.append(f"- ⭐ Stars: {metrics['stars']}")
            if "forks" in metrics:
                report.append(f"- 🍴 Forks: {metrics['forks']}")
            if "commits_count" in metrics:
                report.append(f"- 📝 Commits (30d): {metrics['commits_count']}")
            if "active_contributors" in metrics:
                report.append(f"- 👥 Active Contributors: {metrics['active_contributors']}")
            report.append("")
        
        # Performance metrics
        if any(k in metrics for k in ["application_uptime", "request_rate_5m"]):
            report.append("### Performance")
            if "application_uptime" in metrics:
                uptime_pct = metrics["application_uptime"] * 100
                report.append(f"- 🟢 Uptime: {uptime_pct:.2f}%")
            if "request_rate_5m" in metrics:
                report.append(f"- 📊 Request Rate: {metrics['request_rate_5m']:.2f}/sec")
            if "response_time_p95_ms" in metrics:
                report.append(f"- ⏱️ Response Time (P95): {metrics['response_time_p95_ms']:.0f}ms")
            if "error_rate_percentage" in metrics:
                report.append(f"- ❌ Error Rate: {metrics['error_rate_percentage']:.2f}%")
            report.append("")
        
        # Cost metrics
        if any(k in metrics for k in ["total_cost_usd", "daily_cost_usd"]):
            report.append("### Cost Analysis")
            if "total_cost_usd" in metrics:
                report.append(f"- 💰 Total Cost: ${metrics['total_cost_usd']:.2f}")
            if "daily_cost_usd" in metrics:
                report.append(f"- 📅 Daily Cost: ${metrics['daily_cost_usd']:.2f}")
            if "avg_cost_per_request" in metrics:
                report.append(f"- 💸 Cost per Request: ${metrics['avg_cost_per_request']:.4f}")
            if "hourly_cost_rate" in metrics:
                report.append(f"- 📈 Hourly Rate: ${metrics['hourly_cost_rate']:.2f}/hour")
            report.append("")
        
        # Code quality
        if any(k.startswith("sonar_") for k in metrics.keys()):
            report.append("### Code Quality")
            if "sonar_coverage" in metrics:
                report.append(f"- 🎯 Test Coverage: {metrics['sonar_coverage']:.1f}%")
            if "sonar_code_smells" in metrics:
                report.append(f"- 🔍 Code Smells: {int(metrics['sonar_code_smells'])}")
            if "sonar_bugs" in metrics:
                report.append(f"- 🐛 Bugs: {int(metrics['sonar_bugs'])}")
            if "sonar_vulnerabilities" in metrics:
                report.append(f"- 🔒 Vulnerabilities: {int(metrics['sonar_vulnerabilities'])}")
            report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        report.append("")
        
        recommendations = []
        if health_scores.get("code_quality", 100) < 80:
            recommendations.append("- Improve test coverage and address code quality issues")
        if health_scores.get("performance", 100) < 80:
            recommendations.append("- Optimize response times and investigate performance bottlenecks")
        if health_scores.get("security", 100) < 80:
            recommendations.append("- Address security vulnerabilities and hotspots")
        if health_scores.get("reliability", 100) < 80:
            recommendations.append("- Improve service reliability and reduce error rates")
        
        if not recommendations:
            recommendations.append("- Continue maintaining excellent project health! 🎉")
        
        report.extend(recommendations)
        report.append("")
        
        return "\n".join(report)

@click.command()
@click.option("--config", default=".github/project-metrics.json", help="Path to metrics configuration file")
@click.option("--output", default="project-health-report.md", help="Output file for the report")
@click.option("--format", type=click.Choice(["json", "markdown", "both"]), default="both", help="Output format")
@click.option("--store-db", is_flag=True, help="Store metrics in database")
@click.option("--verbose", is_flag=True, help="Verbose output")
async def main(config: str, output: str, format: str, store_db: bool, verbose: bool):
    """Collect and report project metrics."""
    if verbose:
        console.print("[blue]Starting metrics collection...[/blue]")
    
    async with MetricsCollector(config) as collector:
        all_metrics = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Collect metrics from all sources
            tasks = [
                ("GitHub", collector.collect_github_metrics()),
                ("Prometheus", collector.collect_prometheus_metrics()),
                ("SonarQube", collector.collect_sonarqube_metrics()),
                ("Database", collector.collect_database_metrics())
            ]
            
            for name, coro in tasks:
                task = progress.add_task(f"Collecting {name} metrics...", total=1)
                try:
                    metrics = await coro
                    all_metrics.update(metrics)
                    progress.update(task, completed=1)
                    if verbose:
                        console.print(f"[green]✓ Collected {len(metrics)} metrics from {name}[/green]")
                except Exception as e:
                    progress.update(task, completed=1)
                    console.print(f"[red]✗ Failed to collect {name} metrics: {e}[/red]")
        
        # Generate health scores
        health_scores = collector.generate_health_score(all_metrics)
        
        # Store metrics if requested
        if store_db:
            await collector.store_metrics(all_metrics)
        
        # Generate report
        if format in ["markdown", "both"]:
            report = collector.generate_report(all_metrics, health_scores)
            with open(output, 'w') as f:
                f.write(report)
            console.print(f"[green]✓ Report saved to {output}[/green]")
        
        if format in ["json", "both"]:
            json_output = output.replace(".md", ".json") if output.endswith(".md") else f"{output}.json"
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": all_metrics,
                "health_scores": health_scores
            }
            with open(json_output, 'w') as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]✓ JSON data saved to {json_output}[/green]")
        
        # Display summary
        console.print("\n[bold]📊 Health Summary[/bold]")
        table = Table()
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")
        
        for category, score in health_scores.items():
            status = "🟢 Excellent" if score >= 80 else "🟡 Good" if score >= 60 else "🔴 Needs Attention"
            table.add_row(category.replace('_', ' ').title(), f"{score:.1f}/100", status)
        
        console.print(table)

if __name__ == "__main__":
    asyncio.run(main())