# Repository Automation Administration Guide

This guide covers the automated systems and processes that maintain repository health, security, and quality for the LLM Cost Tracker project.

## Overview

The automation system includes:

1. **Metrics Collection** - Automated project health monitoring
2. **Repository Maintenance** - Dependency updates, security scans, quality checks
3. **Issue Management** - Automated issue creation and triage
4. **Documentation Updates** - Keeping documentation current
5. **Performance Monitoring** - Continuous performance tracking
6. **Security Scanning** - Automated vulnerability detection

## Automation Scripts

### Metrics Collector (`scripts/metrics-collector.py`)

Collects comprehensive project metrics from multiple sources.

#### Features
- **GitHub Metrics**: Stars, forks, commits, contributors, pull requests
- **Prometheus Metrics**: Uptime, request rates, error rates, response times
- **SonarQube Integration**: Code quality, coverage, technical debt
- **Database Metrics**: Application-specific cost and usage data
- **Health Scoring**: Overall project health assessment
- **Report Generation**: Markdown and JSON reports

#### Usage
```bash
# Basic metrics collection
python scripts/metrics-collector.py

# Store metrics in database
python scripts/metrics-collector.py --store-db

# Generate JSON output
python scripts/metrics-collector.py --format json

# Verbose output
python scripts/metrics-collector.py --verbose
```

#### Configuration
Metrics collection is configured in `.github/project-metrics.json`:

```json
{
  "tracking": {
    "data_sources": {
      "github": {
        "api_endpoint": "https://api.github.com",
        "metrics": ["commits_per_week", "pull_requests_merged"]
      },
      "prometheus": {
        "endpoint": "http://prometheus:9090",
        "metrics": ["application_uptime", "request_rate"]
      }
    }
  }
}
```

### Repository Automation (`scripts/repository-automation.py`)

Performs automated maintenance tasks.

#### Features
- **Dependency Updates**: Automated package updates with safety checks
- **Security Scanning**: Vulnerability detection and issue creation
- **Code Quality Analysis**: Coverage and complexity monitoring
- **Documentation Maintenance**: README badges, changelog updates
- **GitHub Integration**: Automated issue and PR creation
- **Slack Notifications**: Team communication

#### Usage
```bash
# Full automation run
python scripts/repository-automation.py

# Dry run (show what would be done)
python scripts/repository-automation.py --dry-run

# Skip specific tasks
python scripts/repository-automation.py --skip-deps --skip-security

# Create GitHub issues for problems found
python scripts/repository-automation.py --create-issues

# Send Slack notifications
python scripts/repository-automation.py --notify-slack
```

## Scheduled Automation

### GitHub Actions Integration

Create `.github/workflows/automation.yml`:

```yaml
name: Repository Automation

on:
  schedule:
    # Run daily at 6 AM UTC
    - cron: '0 6 * * *'
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Run in dry-run mode'
        required: false
        default: 'false'
        type: boolean

jobs:
  automation:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      issues: write
      pull-requests: write
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install aiohttp asyncpg click rich
          poetry install
          
      - name: Run metrics collection
        run: |
          python scripts/metrics-collector.py --store-db --verbose
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          
      - name: Run repository automation
        run: |
          python scripts/repository-automation.py \
            ${{ github.event.inputs.dry_run == 'true' && '--dry-run' || '' }} \
            --create-issues \
            --notify-slack \
            --verbose
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
          
      - name: Upload automation reports
        uses: actions/upload-artifact@v3
        with:
          name: automation-reports
          path: |
            project-health-report.*
            automation-report-*.md
```

### Cron Jobs (Alternative)

For server-based deployments, set up cron jobs:

```bash
# Add to crontab
crontab -e

# Daily metrics collection at 6 AM
0 6 * * * cd /path/to/llm-cost-tracker && /usr/bin/python3 scripts/metrics-collector.py --store-db

# Weekly automation run on Sundays at 8 AM
0 8 * * 0 cd /path/to/llm-cost-tracker && /usr/bin/python3 scripts/repository-automation.py --create-issues --notify-slack

# Monthly comprehensive report on 1st of month at 9 AM
0 9 1 * * cd /path/to/llm-cost-tracker && /usr/bin/python3 scripts/metrics-collector.py --format both --store-db
```

## Metrics Configuration

### Health Score Calculation

The system calculates health scores based on multiple factors:

#### Code Quality Score (0-100)
```python
def calculate_code_quality_score(metrics):
    base_score = min(100, max(0, metrics.get("coverage", 0)))
    
    # Bonus for good ratings
    if "sonar_maintainability_rating" in metrics:
        rating_bonus = max(0, (6 - metrics["sonar_maintainability_rating"]) * 5)
        base_score = min(100, base_score + rating_bonus)
    
    return base_score
```

#### Reliability Score (0-100)
```python
def calculate_reliability_score(metrics):
    uptime = metrics.get("application_uptime", 0) * 100
    error_rate = metrics.get("error_rate_percentage", 100)
    return min(100, max(0, uptime - error_rate))
```

#### Performance Score (0-100)
```python
def calculate_performance_score(metrics):
    response_time = metrics.get("response_time_p95_ms", 5000)
    threshold = 2000  # 2 second threshold
    return max(0, min(100, 100 - (response_time / threshold * 100)))
```

#### Security Score (0-100)
```python
def calculate_security_score(metrics):
    vulnerabilities = metrics.get("sonar_vulnerabilities", 0)
    hotspots = metrics.get("sonar_security_hotspots", 0)
    return max(0, 100 - (vulnerabilities * 10 + hotspots * 5))
```

### Alert Thresholds

Configure alert thresholds in `.github/project-metrics.json`:

```json
{
  "alerts": {
    "critical": {
      "conditions": [
        "security.critical_vulnerabilities > 0",
        "reliability.uptime < 99.0",
        "performance.response_time_p99 > 10000"
      ],
      "notification_channels": ["pagerduty", "slack-critical"],
      "escalation_time_minutes": 15
    },
    "warning": {
      "conditions": [
        "code_quality.coverage < 80",
        "security.high_vulnerabilities > 5",
        "performance.build_time > 15"
      ],
      "notification_channels": ["slack-alerts", "email"],
      "escalation_time_minutes": 60
    }
  }
}
```

## Automation Policies

### Dependency Updates

Configure automatic dependency updates:

```json
{
  "automation": {
    "dependency_updates": {
      "enabled": true,
      "frequency": "weekly",
      "auto_merge": {
        "patch": true,
        "minor": false,
        "major": false
      },
      "security_updates": {
        "auto_merge": true,
        "notification_channels": ["slack-security"]
      }
    }
  }
}
```

### Security Scanning

Configure security scanning policies:

```json
{
  "automation": {
    "security_scanning": {
      "enabled": true,
      "frequency": "daily",
      "tools": ["bandit", "safety", "semgrep", "trivy"],
      "fail_on_high": true,
      "fail_on_critical": true,
      "auto_create_issues": true,
      "issue_labels": ["security", "vulnerability", "automation"]
    }
  }
}
```

### Code Quality Gates

Set up quality gates:

```json
{
  "automation": {
    "code_quality": {
      "enabled": true,
      "quality_gates": {
        "coverage_minimum": 85,
        "maintainability_rating": "A",
        "reliability_rating": "A",
        "security_rating": "A",
        "complexity_threshold": 10
      },
      "auto_create_issues": true,
      "improvement_suggestions": true
    }
  }
}
```

## Integration Setup

### Slack Integration

1. Create a Slack app or webhook
2. Add webhook URL to repository secrets as `SLACK_WEBHOOK_URL`
3. Configure notification channels:

```json
{
  "integrations": {
    "slack": {
      "enabled": true,
      "channels": {
        "alerts": "#alerts",
        "deployments": "#deployments",
        "metrics": "#metrics",
        "security": "#security"
      }
    }
  }
}
```

### SonarQube Integration

1. Set up SonarCloud project
2. Add `SONAR_TOKEN` to repository secrets
3. Configure project key in metrics configuration

### Database Integration

1. Set up PostgreSQL database
2. Add `DATABASE_URL` to repository secrets
3. Create metrics storage table:

```sql
CREATE TABLE project_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC,
    metric_metadata JSONB,
    source VARCHAR(100)
);

CREATE INDEX idx_project_metrics_timestamp ON project_metrics(timestamp);
CREATE INDEX idx_project_metrics_name ON project_metrics(metric_name);
CREATE INDEX idx_project_metrics_source ON project_metrics(source);
```

## Monitoring and Alerting

### Prometheus Metrics

The automation system exposes its own metrics:

```python
# Automation execution metrics
automation_runs_total = Counter('automation_runs_total', 'Total automation runs')
automation_duration_seconds = Histogram('automation_duration_seconds', 'Automation execution time')
automation_changes_total = Counter('automation_changes_total', 'Changes made by automation', ['type'])
automation_issues_created_total = Counter('automation_issues_created_total', 'Issues created by automation')

# Health score metrics
project_health_score = Gauge('project_health_score', 'Overall project health score', ['category'])
```

### Grafana Dashboards

Create dashboards for automation monitoring:

#### Automation Overview Panel
```json
{
  "title": "Automation Status",
  "type": "stat",
  "targets": [
    {
      "expr": "automation_runs_total",
      "legendFormat": "Total Runs"
    },
    {
      "expr": "rate(automation_changes_total[24h])",
      "legendFormat": "Daily Changes"
    }
  ]
}
```

#### Health Score Panel
```json
{
  "title": "Project Health Scores",
  "type": "gauge",
  "targets": [
    {
      "expr": "project_health_score",
      "legendFormat": "{{category}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "min": 0,
      "max": 100,
      "thresholds": {
        "steps": [
          {"color": "red", "value": 0},
          {"color": "yellow", "value": 60},
          {"color": "green", "value": 80}
        ]
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

#### Metrics Collection Failures

**Problem**: Metrics collector fails to fetch data
**Solutions**:
- Verify API tokens are valid and have correct permissions
- Check network connectivity to external services
- Review rate limiting and API quotas
- Validate configuration file syntax

#### Automation Script Errors

**Problem**: Repository automation script fails
**Solutions**:
- Check Python dependencies are installed
- Verify GitHub token has necessary permissions
- Review file permissions for script execution
- Check for conflicting automation runs

#### Database Connection Issues

**Problem**: Cannot store metrics in database
**Solutions**:
- Verify DATABASE_URL is correct
- Check database connectivity and permissions
- Ensure database schema exists
- Review connection pool settings

### Debug Commands

```bash
# Test metrics collection without storing
python scripts/metrics-collector.py --format json --verbose

# Run automation in dry-run mode
python scripts/repository-automation.py --dry-run --verbose

# Test individual components
python -c "
import asyncio
from scripts.metrics_collector import MetricsCollector

async def test():
    async with MetricsCollector() as collector:
        metrics = await collector.collect_github_metrics()
        print(f'Collected {len(metrics)} GitHub metrics')

asyncio.run(test())
"

# Validate configuration
python -c "
import json
with open('.github/project-metrics.json') as f:
    config = json.load(f)
print('Configuration valid')
"
```

### Log Analysis

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler()
    ]
)
```

## Performance Optimization

### Caching Strategies

```python
# Cache API responses
import aiohttp_cache
from aiohttp_cache import cache

session = aiohttp_cache.CachedSession(
    cache=cache.RedisCache(),
    expire_after=300  # 5 minutes
)
```

### Parallel Processing

```python
# Collect metrics in parallel
async def collect_all_metrics():
    tasks = [
        collect_github_metrics(),
        collect_prometheus_metrics(),
        collect_sonarqube_metrics(),
        collect_database_metrics()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Database Optimization

```sql
-- Optimize metrics table
CREATE INDEX CONCURRENTLY idx_project_metrics_timestamp 
ON project_metrics(timestamp DESC);

-- Partition by month for large datasets
CREATE TABLE project_metrics_202401 
PARTITION OF project_metrics 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## Maintenance Schedule

### Daily Tasks
- Metrics collection and health scoring
- Security vulnerability scanning
- Basic automation checks

### Weekly Tasks
- Dependency update reviews
- Code quality analysis
- Documentation updates
- Performance trend analysis

### Monthly Tasks
- Comprehensive health reports
- Automation system review
- Configuration updates
- Long-term trend analysis

### Quarterly Tasks
- Automation strategy review
- Tool evaluation and updates
- Process optimization
- Security audit of automation systems

This comprehensive automation system ensures continuous repository health monitoring and maintenance while reducing manual overhead and improving project quality.