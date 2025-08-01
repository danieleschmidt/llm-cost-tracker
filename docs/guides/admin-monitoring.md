# Monitoring and Observability Administration Guide

This guide covers the complete monitoring and observability setup for LLM Cost Tracker administrators.

## Architecture Overview

The monitoring stack consists of:

1. **OpenTelemetry Collector** - Metrics and traces collection
2. **Prometheus** - Metrics storage and alerting
3. **Grafana** - Visualization and dashboards  
4. **AlertManager** - Alert routing and notifications
5. **Custom Metrics** - Application-specific LLM cost metrics

## Metrics Collection

### Application Metrics

The LLM Cost Tracker exposes the following metric categories:

#### Cost Metrics
```
llm_cost_tracker_total_cost_usd - Total cost accumulated by model/user/application
llm_cost_tracker_cost_per_request - Cost per individual request
llm_cost_tracker_hourly_cost_rate - Current hourly cost rate
llm_cost_tracker_daily_budget_utilization - Percentage of daily budget used
```

#### Performance Metrics
```
llm_cost_tracker_request_duration_seconds - Request latency histogram
llm_cost_tracker_requests_total - Total requests by model/status
llm_cost_tracker_errors_total - Error count by type
llm_cost_tracker_concurrent_requests - Active concurrent requests
```

#### Token Usage Metrics
```
llm_cost_tracker_prompt_tokens_total - Prompt tokens consumed
llm_cost_tracker_completion_tokens_total - Completion tokens generated
llm_cost_tracker_tokens_per_request - Token usage distribution
```

#### System Metrics
```
llm_cost_tracker_memory_usage_bytes - Application memory usage
llm_cost_tracker_cpu_usage_ratio - CPU utilization
llm_cost_tracker_database_connections - Active DB connections
llm_cost_tracker_cache_hit_ratio - Cache performance
```

### Custom Recording Rules

Located in `config/recording-rules.yml`:

```yaml
groups:
- name: llm_cost_recording_rules
  interval: 30s
  rules:
  # Cost rates
  - record: llm:cost_per_model_1h
    expr: increase(llm_cost_tracker_total_cost_usd[1h])
    labels:
      aggregation: "1h"

  - record: llm:cost_per_model_24h  
    expr: increase(llm_cost_tracker_total_cost_usd[24h])
    labels:
      aggregation: "24h"

  # Request rates
  - record: llm:request_rate_5m
    expr: rate(llm_cost_tracker_requests_total[5m])

  - record: llm:error_rate_5m
    expr: rate(llm_cost_tracker_errors_total[5m]) / rate(llm_cost_tracker_requests_total[5m])

  # Latency percentiles
  - record: llm:latency_p50_5m
    expr: histogram_quantile(0.50, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))

  - record: llm:latency_p95_5m
    expr: histogram_quantile(0.95, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))

  - record: llm:latency_p99_5m
    expr: histogram_quantile(0.99, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))

  # Budget utilization
  - record: budget:utilization_percentage
    expr: |
      (llm:cost_per_model_24h / on(application) group_left() 
       llm_cost_tracker_daily_budget_limit) * 100

  - record: budget:days_remaining
    expr: |
      (llm_cost_tracker_monthly_budget_limit - llm:cost_per_model_24h * 30) / 
      llm:cost_per_model_24h

  # Efficiency metrics
  - record: llm:cost_per_1k_tokens
    expr: |
      llm_cost_tracker_total_cost_usd * 1000 / 
      (llm_cost_tracker_prompt_tokens_total + llm_cost_tracker_completion_tokens_total)

  - record: llm:tokens_per_request
    expr: |
      (llm_cost_tracker_prompt_tokens_total + llm_cost_tracker_completion_tokens_total) / 
      llm_cost_tracker_requests_total
```

## Dashboard Configuration

### Main LLM Cost Dashboard

Key panels and queries:

#### Cost Overview Panel
```promql
# Total daily cost
sum(llm:cost_per_model_24h)

# Cost by model
sum(llm:cost_per_model_1h) by (model)

# Cost trend over time
llm:cost_per_model_1h

# Budget utilization
budget:utilization_percentage
```

#### Performance Overview Panel
```promql
# Request rate
sum(llm:request_rate_5m)

# Error rate
sum(llm:error_rate_5m) * 100

# 95th percentile latency
llm:latency_p95_5m

# Concurrent requests
llm_cost_tracker_concurrent_requests
```

#### Usage Analytics Panel
```promql
# Top applications by cost
topk(10, sum(llm:cost_per_model_1h) by (application))

# Top users by requests
topk(10, sum(llm:request_rate_5m) by (user_id))

# Model efficiency comparison
llm:cost_per_1k_tokens

# Token usage patterns
rate(llm_cost_tracker_prompt_tokens_total[5m])
```

### System Health Dashboard

#### Infrastructure Metrics
```promql
# Memory usage
llm_cost_tracker_memory_usage_bytes / 1024 / 1024 / 1024

# CPU usage
rate(llm_cost_tracker_cpu_usage_ratio[5m]) * 100

# Database connections
llm_cost_tracker_database_connections

# Disk usage
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100
```

#### Service Health
```promql
# Service uptime
up{job="llm-cost-tracker"}

# Health check success rate
rate(llm_cost_tracker_health_checks_total{status="success"}[5m])

# Cache hit ratio
llm_cost_tracker_cache_hit_ratio * 100
```

## Alert Configuration

### Alert Routing

Configure AlertManager (`config/alertmanager.yml`):

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@terragonlabs.com'
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 5s
    repeat_interval: 15m
  - match:
      category: cost
    receiver: 'cost-alerts'
  - match:
      category: security
    receiver: 'security-alerts'

receivers:
- name: 'default'
  slack_configs:
  - channel: '#alerts'
    title: 'LLM Cost Tracker Alert'
    text: |
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Severity:* {{ .Labels.severity }}
      {{ end }}

- name: 'critical-alerts'
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_KEY'
    description: |
      {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
  slack_configs:
  - channel: '#critical-alerts'
    title: '🚨 CRITICAL: LLM Cost Tracker'
    color: 'danger'

- name: 'cost-alerts'
  slack_configs:
  - channel: '#cost-optimization'
    title: '💰 Cost Alert'
    color: 'warning'
  email_configs:
  - to: 'finance@terragonlabs.com'
    subject: 'LLM Cost Alert: {{ .GroupLabels.alertname }}'

- name: 'security-alerts'
  slack_configs:
  - channel: '#security-incidents'
    title: '🔒 Security Alert'
    color: 'danger'
  email_configs:
  - to: 'security@terragonlabs.com'
    subject: 'Security Alert: {{ .GroupLabels.alertname }}'
```

### Custom Alert Rules

#### Business Logic Alerts
```yaml
- alert: ModelCostEfficiencyDegraded
  expr: |
    (
      llm:cost_per_1k_tokens > 
      (avg_over_time(llm:cost_per_1k_tokens[7d]) * 1.5)
    ) and llm:cost_per_1k_tokens > 0.01
  for: 15m
  labels:
    severity: warning
    category: efficiency
  annotations:
    summary: "Model cost efficiency has degraded"
    description: "Cost per 1k tokens for {{ $labels.model }} is {{ $value | printf \"%.4f\" }}, 50% above weekly average"

- alert: UnusualTokenUsagePattern
  expr: |
    abs(
      llm:tokens_per_request - 
      avg_over_time(llm:tokens_per_request[24h])
    ) > (stddev_over_time(llm:tokens_per_request[24h]) * 3)
  for: 10m
  labels:
    severity: info
    category: anomaly
  annotations:
    summary: "Unusual token usage pattern detected"
    description: "Token usage for {{ $labels.application }} deviates significantly from normal patterns"
```

#### Predictive Alerts
```yaml
- alert: BudgetProjectedToExceed
  expr: |
    predict_linear(llm:cost_per_model_24h[4h], 24*3600) > 
    on(application) llm_cost_tracker_daily_budget_limit
  for: 30m
  labels:
    severity: warning
    category: forecast
  annotations:
    summary: "Daily budget projected to be exceeded"
    description: "Based on current trend, {{ $labels.application }} will exceed daily budget"

- alert: MonthlyBudgetTrendConcern
  expr: |
    predict_linear(llm:cost_per_model_24h[7d], 30*24*3600) > 
    on(application) llm_cost_tracker_monthly_budget_limit * 0.8
  for: 1h
  labels:
    severity: info
    category: forecast
  annotations:
    summary: "Monthly budget trend shows potential overrun"
    description: "Current 7-day trend suggests {{ $labels.application }} may exceed 80% of monthly budget"
```

## Log Management

### Structured Logging Configuration

The application uses structured logging with these levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about application flow
- **WARNING**: Something unexpected happened but application continues
- **ERROR**: Serious problem that needs attention
- **CRITICAL**: Very serious error that may cause application to abort

### Log Aggregation Setup

#### Using ELK Stack
```yaml
# docker-compose-logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./config/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

#### Logstash Configuration (`config/logstash.conf`)
```ruby
input {
  tcp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [logger_name] {
    mutate {
      add_field => { "service" => "llm-cost-tracker" }
    }
  }

  # Parse cost-related logs
  if [message] =~ /cost.*\$/ {
    grok {
      match => { "message" => "cost.*\$(?<cost_amount>[0-9.]+)" }
    }
    mutate {
      convert => { "cost_amount" => "float" }
      add_tag => [ "cost_event" ]
    }
  }

  # Parse performance logs
  if [message] =~ /latency.*ms/ {
    grok {
      match => { "message" => "latency.*(?<latency_ms>[0-9.]+)ms" }
    }
    mutate {
      convert => { "latency_ms" => "float" }
      add_tag => [ "performance_event" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "llm-cost-tracker-%{+YYYY.MM.dd}"
  }
}
```

### Log Analysis Queries

#### Cost Analysis
```json
{
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-1d"}}},
        {"exists": {"field": "cost_amount"}}
      ]
    }
  },
  "aggs": {
    "cost_by_model": {
      "terms": {"field": "model.keyword"},
      "aggs": {
        "total_cost": {"sum": {"field": "cost_amount"}}
      }
    }
  }
}
```

#### Error Analysis
```json
{
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-4h"}}},
        {"term": {"level": "ERROR"}}
      ]
    }
  },
  "aggs": {
    "errors_by_type": {
      "terms": {"field": "exception.class.keyword"}
    }
  }
}
```

## Health Checks

### Application Health Endpoints

The application exposes several health check endpoints:

#### Basic Health Check
```bash
curl http://llm-cost-tracker:8000/health
# Response: {"status": "healthy", "timestamp": "2024-01-15T10:30:00Z"}
```

#### Detailed Health Check
```bash
curl http://llm-cost-tracker:8000/health/detailed
# Response includes database, external APIs, and internal component status
```

#### Component-Specific Checks
```bash
# Database connectivity
curl http://llm-cost-tracker:8000/health/database

# External API health (LLM providers)
curl http://llm-cost-tracker:8000/health/providers

# Cache status
curl http://llm-cost-tracker:8000/health/cache

# Budget system status
curl http://llm-cost-tracker:8000/health/budget-engine
```

### Health Check Automation

#### Kubernetes Liveness/Readiness Probes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-cost-tracker
spec:
  template:
    spec:
      containers:
      - name: llm-cost-tracker
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

#### Prometheus Health Check Monitoring
```yaml
- alert: HealthCheckFailing
  expr: llm_cost_tracker_health_check_success_ratio < 0.95
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Health checks are failing"
    description: "Health check success rate is {{ $value | printf \"%.2f\" }}%"
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

#### Response Time SLIs
- **95th percentile latency < 2 seconds**
- **99th percentile latency < 5 seconds**
- **Average response time < 500ms**

#### Availability SLIs
- **Service uptime > 99.9%**
- **Health check success rate > 99.5%**
- **Error rate < 0.1%**

#### Cost Efficiency SLIs
- **Cost per 1k tokens within 10% of baseline**
- **Budget utilization alerts < 5 per day**
- **Model switching effectiveness > 20% cost reduction**

### Performance Dashboards

#### Latency Analysis
```promql
# Request latency percentiles
histogram_quantile(0.50, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(llm_cost_tracker_request_duration_seconds_bucket[5m]))

# Latency by model
histogram_quantile(0.95, rate(llm_cost_tracker_request_duration_seconds_bucket[5m])) by (model)

# Latency trend over time
rate(llm_cost_tracker_request_duration_seconds_sum[5m]) / rate(llm_cost_tracker_request_duration_seconds_count[5m])
```

#### Throughput Analysis
```promql
# Requests per second
rate(llm_cost_tracker_requests_total[5m])

# Concurrent requests
llm_cost_tracker_concurrent_requests

# Request queue depth
llm_cost_tracker_request_queue_depth
```

## Capacity Planning

### Resource Usage Monitoring

#### Memory Analysis
```promql
# Memory usage trend
llm_cost_tracker_memory_usage_bytes

# Memory usage by component
llm_cost_tracker_memory_usage_bytes by (component)

# Memory growth rate
increase(llm_cost_tracker_memory_usage_bytes[1h])
```

#### CPU Analysis
```promql
# CPU utilization
rate(llm_cost_tracker_cpu_usage_ratio[5m]) * 100

# CPU usage by operation
rate(llm_cost_tracker_cpu_usage_ratio[5m]) by (operation)
```

#### Database Capacity
```promql  
# Database size growth
increase(llm_cost_tracker_database_size_bytes[24h])

# Connection pool utilization
llm_cost_tracker_database_connections / llm_cost_tracker_database_max_connections * 100

# Query performance
rate(llm_cost_tracker_database_query_duration_seconds_sum[5m]) / rate(llm_cost_tracker_database_query_duration_seconds_count[5m])
```

### Scaling Recommendations

#### Horizontal Scaling Triggers
```yaml
- alert: HighCPULoadSustained
  expr: rate(llm_cost_tracker_cpu_usage_ratio[5m]) > 0.70
  for: 10m
  annotations:
    summary: "Consider horizontal scaling"
    action: "Add more application replicas"

- alert: HighMemoryPressure
  expr: llm_cost_tracker_memory_usage_bytes / llm_cost_tracker_memory_limit_bytes > 0.85
  for: 5m
  annotations:
    summary: "Memory pressure detected"
    action: "Scale up instance size or add replicas"
```

#### Database Scaling Indicators
```yaml
- alert: DatabaseConnectionsHigh
  expr: llm_cost_tracker_database_connections / llm_cost_tracker_database_max_connections > 0.80
  for: 5m
  annotations:
    summary: "Database connection pool utilization high"
    action: "Consider read replicas or connection pooling optimization"

- alert: DatabaseSlowQueries
  expr: rate(llm_cost_tracker_database_slow_queries_total[5m]) > 0.1
  for: 10m
  annotations:
    summary: "Database performance degradation"
    action: "Optimize queries or scale database resources"
```

## Troubleshooting

### Common Monitoring Issues

#### Missing Metrics
1. **Check metric exposition**: `curl http://llm-cost-tracker:8000/metrics | grep llm_cost`
2. **Verify Prometheus scraping**: Check Prometheus targets page
3. **Validate metric names**: Ensure consistent naming conventions
4. **Check service discovery**: Verify Kubernetes service annotations

#### Alert Fatigue
1. **Review alert thresholds**: Adjust based on normal operating ranges
2. **Implement alert grouping**: Group related alerts to reduce noise
3. **Add alert context**: Include runbook links and action items
4. **Regular alert tuning**: Monthly review of firing alerts

#### Dashboard Performance
1. **Optimize queries**: Use recording rules for complex calculations
2. **Reduce time ranges**: Limit default dashboard time windows
3. **Implement caching**: Use Grafana query caching features
4. **Query optimization**: Use efficient PromQL patterns

### Debug Tools

#### Prometheus Query Testing
```bash
# Test queries directly
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=llm_cost_tracker_requests_total'

# Query range for time series
curl -G http://prometheus:9090/api/v1/query_range \
  --data-urlencode 'query=rate(llm_cost_tracker_requests_total[5m])' \
  --data-urlencode 'start=2024-01-15T00:00:00Z' \
  --data-urlencode 'end=2024-01-15T12:00:00Z' \
  --data-urlencode 'step=300'
```

#### Grafana API Usage
```bash
# Get dashboard JSON
curl -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
  http://grafana:3000/api/dashboards/uid/llm-cost-dashboard

# Create alert via API
curl -X POST \
  -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @new-alert.json \
  http://grafana:3000/api/alerts
```

This comprehensive monitoring guide provides the foundation for maintaining high visibility into LLM Cost Tracker operations, enabling proactive issue resolution and cost optimization.