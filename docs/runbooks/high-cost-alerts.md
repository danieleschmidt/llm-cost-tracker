# High Cost Alerts Runbook

## Overview

This runbook covers responding to LLM cost threshold alerts, including daily cost limits, budget overruns, and unexpected cost spikes.

## Alert Types

### HighDailyCost
- **Threshold**: Daily costs > $100 for any model
- **Severity**: Warning
- **Response Time**: 30 minutes

### CriticalDailyCost  
- **Threshold**: Daily costs > $500 for any model
- **Severity**: Critical
- **Response Time**: 5 minutes

### BudgetThresholdExceeded
- **Threshold**: Budget utilization > 90%
- **Severity**: Critical
- **Response Time**: 10 minutes

### UnexpectedCostSpike
- **Threshold**: Hourly cost > 3x normal average
- **Severity**: Warning
- **Response Time**: 15 minutes

## Immediate Actions

### 1. Assess Impact (2 minutes)
```bash
# Check current cost rate
curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:hourly_cost_rate | jq '.data.result[0].value[1]'

# Check which applications are driving costs
curl -s http://prometheus:9090/api/v1/query?query=topk\(5,sum\(llm_cost_tracker_total_cost_usd\)by\(application\)\) | jq
```

### 2. Identify Top Cost Drivers (3 minutes)
```bash
# Top models by cost
curl -s http://prometheus:9090/api/v1/query?query=topk\(5,sum\(llm_cost_tracker_total_cost_usd\)by\(model\)\) | jq

# Top users by cost (if user tracking enabled)  
curl -s http://prometheus:9090/api/v1/query?query=topk\(10,sum\(llm_cost_tracker_total_cost_usd\)by\(user_id\)\) | jq

# Cost by time period
curl -s "http://prometheus:9090/api/v1/query_range?query=llm_cost_tracker:hourly_cost_rate&start=$(date -d '24 hours ago' -u +%s)&end=$(date -u +%s)&step=3600" | jq
```

### 3. Check for Anomalies (2 minutes)
```bash
# Check error rates (high errors might indicate retry loops)
curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:error_rate_5m | jq

# Check request patterns
curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:request_rate_5m | jq

# Check for specific application issues
kubectl logs -n llm-cost-tracker deployment/llm-cost-tracker --tail=100 | grep -E "(ERROR|WARN)"
```

## Investigation Steps

### 1. Analyze Cost Patterns

#### View Grafana Dashboard
1. Open [LLM Cost Dashboard](http://grafana:3000/d/llm-costs)
2. Set time range to last 24 hours
3. Examine:
   - Cost per model over time
   - Request volume correlation
   - User/application breakdown
   - Token usage patterns

#### Query Historical Data
```bash
# Cost trend over past week
curl -s "http://prometheus:9090/api/v1/query_range?query=llm_cost_tracker:daily_cost_rate&start=$(date -d '7 days ago' -u +%s)&end=$(date -u +%s)&step=86400" | jq '.data.result[0].values' | jq -r '.[] | .[0] + " " + .[1]' | while read timestamp value; do echo "$(date -d @$timestamp '+%Y-%m-%d'): \$$(printf '%.2f' $value)"; done
```

### 2. Check Application Logs

#### Application-Specific Investigation
```bash
# Check for specific applications with high usage
kubectl logs -n llm-cost-tracker deployment/llm-cost-tracker -c llm-cost-tracker | grep -E "high_cost|budget_exceeded" | tail -20

# Check for retry patterns or loops
kubectl logs -n llm-cost-tracker deployment/llm-cost-tracker -c llm-cost-tracker | grep -c "retry" | head -100

# Look for specific user patterns
kubectl logs -n llm-cost-tracker deployment/llm-cost-tracker -c llm-cost-tracker | grep "user_id" | cut -d'"' -f4 | sort | uniq -c | sort -nr | head -10
```

#### Database Investigation
```bash
# Connect to database
kubectl exec -n llm-cost-tracker deployment/postgres -it -- psql -U postgres -d llm_metrics

# Query top cost drivers
psql> SELECT 
  model, 
  COUNT(*) as requests,
  SUM(cost_usd) as total_cost,
  AVG(cost_usd) as avg_cost,
  SUM(total_tokens) as total_tokens
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY model 
ORDER BY total_cost DESC 
LIMIT 10;

# Query by application
psql> SELECT 
  application,
  COUNT(*) as requests,
  SUM(cost_usd) as total_cost
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY application 
ORDER BY total_cost DESC;

# Query hourly breakdown
psql> SELECT 
  DATE_TRUNC('hour', timestamp) as hour,
  SUM(cost_usd) as hourly_cost,
  COUNT(*) as requests
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY hour 
ORDER BY hour DESC;
```

### 3. Check External Factors

#### LLM Provider Status
```bash
# Check OpenAI status
curl -s https://status.openai.com/api/v2/status.json | jq '.status.description'

# Check Anthropic status  
curl -s https://status.anthropic.com/api/v2/status.json | jq '.status.description'

# Check our provider API health
curl -f http://llm-cost-tracker:8000/health/providers || echo "Provider health check failed"
```

## Resolution Steps

### For Budget Threshold Exceeded

#### 1. Enable Cost Controls (Immediate - 2 minutes)
```bash
# Enable automatic model swapping
curl -X POST http://llm-cost-tracker:8000/api/v1/config/budget \
  -H "Content-Type: application/json" \
  -d '{"enable_model_swapping": true, "swap_threshold": 0.85}'

# Enable rate limiting
curl -X POST http://llm-cost-tracker:8000/api/v1/config/rate-limit \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "requests_per_minute": 100}'
```

#### 2. Implement Circuit Breakers (5 minutes)
```bash
# Set emergency cost limits
kubectl patch configmap llm-cost-tracker-config -n llm-cost-tracker --patch='
data:
  EMERGENCY_COST_LIMIT: "1000"
  ENABLE_EMERGENCY_STOP: "true"
'

# Restart deployment to pick up new config
kubectl rollout restart deployment/llm-cost-tracker -n llm-cost-tracker
```

#### 3. Notify Stakeholders (3 minutes)
```bash
# Send Slack notification
curl -X POST -H 'Content-type: application/json' \
--data '{
  "text":"🚨 LLM Cost Alert: Budget threshold exceeded",
  "blocks":[
    {
      "type":"section",
      "text":{
        "type":"mrkdwn",
        "text":"*Cost Alert*\nBudget utilization: 90%+\nCurrent daily rate: $'$(curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:daily_cost_rate | jq -r '.data.result[0].value[1]')'\n\n*Actions Taken:*\n• Enabled model swapping\n• Implemented rate limiting\n• Activated emergency stops"
      }
    }
  ]
}' \
$SLACK_WEBHOOK_URL
```

### For Unexpected Cost Spike

#### 1. Identify Spike Source (5 minutes)
```bash
# Get applications with highest increase
curl -s "http://prometheus:9090/api/v1/query?query=increase(llm_cost_tracker_total_cost_usd[1h])" | jq -r '.data.result[] | "\(.metric.application): $\(.value[1])"' | sort -rn

# Check for specific users driving spike
curl -s "http://prometheus:9090/api/v1/query?query=topk(5,increase(llm_cost_tracker_total_cost_usd[1h]))" | jq
```

#### 2. Investigate Potential Abuse (10 minutes)
```bash
# Check for unusual request patterns
kubectl logs -n llm-cost-tracker deployment/llm-cost-tracker | grep -E "user_id|api_key" | tail -100 | awk '{print $5}' | sort | uniq -c | sort -nr | head -10

# Check token usage patterns
psql -U postgres -d llm_metrics -c "
SELECT 
  user_id,
  AVG(total_tokens) as avg_tokens,
  MAX(total_tokens) as max_tokens,
  COUNT(*) as requests
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY user_id
HAVING COUNT(*) > 10
ORDER BY avg_tokens DESC
LIMIT 10;"
```

#### 3. Implement Targeted Controls (5 minutes)
```bash
# Rate limit specific users if needed
curl -X POST http://llm-cost-tracker:8000/api/v1/admin/rate-limit \
  -H "Content-Type: application/json" \
  -d '{"user_id": "suspicious_user_id", "requests_per_minute": 10}'

# Block abusive patterns
curl -X POST http://llm-cost-tracker:8000/api/v1/admin/block \
  -H "Content-Type: application/json" \
  -d '{"pattern": "abuse_pattern", "duration": "1h"}'
```

### For Daily Cost Exceeded

#### 1. Assess Current Trajectory (2 minutes)
```bash
# Calculate projected daily total
current_hour=$(date +%H)
current_cost=$(curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:daily_cost_rate | jq -r '.data.result[0].value[1]')
hourly_rate=$(curl -s http://prometheus:9090/api/v1/query?query=llm_cost_tracker:hourly_cost_rate | jq -r '.data.result[0].value[1]')

projected_daily=$(echo "$current_cost + ($hourly_rate * (24 - $current_hour))" | bc -l)
echo "Current: \$$current_cost, Projected: \$$projected_daily"
```

#### 2. Optimize Model Usage (10 minutes)
```bash
# Switch expensive models to cheaper alternatives
curl -X POST http://llm-cost-tracker:8000/api/v1/config/model-mapping \
  -H "Content-Type: application/json" \
  -d '{
    "mappings": [
      {"from": "gpt-4", "to": "gpt-3.5-turbo", "condition": "cost_over_threshold"},
      {"from": "claude-3-opus", "to": "claude-3-sonnet", "condition": "cost_over_threshold"}
    ]
  }'

# Adjust model parameters for cost optimization
curl -X POST http://llm-cost-tracker:8000/api/v1/config/model-params \
  -H "Content-Type: application/json" \
  -d '{
    "max_tokens": 1000,
    "temperature": 0.7,
    "cost_optimization": true
  }'
```

## Prevention Measures

### 1. Budget Monitoring
```bash
# Set up proactive budget alerts at 50%, 70%, 85%
curl -X POST http://llm-cost-tracker:8000/api/v1/config/budget-alerts \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": [0.5, 0.7, 0.85],
    "notifications": ["slack", "email"],
    "actions": ["warn", "model_swap", "rate_limit"]
  }'
```

### 2. Usage Patterns
```bash
# Implement usage quotas per application
curl -X POST http://llm-cost-tracker:8000/api/v1/config/quotas \
  -H "Content-Type: application/json" \
  -d '{
    "daily_limits": {
      "default": 100,
      "high_priority": 500,
      "batch_processing": 1000
    }
  }'
```

### 3. Monitoring Improvements
- Set up anomaly detection for cost patterns
- Implement predictive cost alerts based on trends  
- Create cost efficiency dashboards for teams
- Regular cost review meetings with stakeholders

## Escalation

### When to Escalate
- Daily costs exceed $1000 
- Cannot identify cost spike source within 30 minutes
- Controls are not reducing cost rate
- Suspected security breach or abuse

### Escalation Steps
1. **Page On-Call Engineer** (for critical alerts)
   ```bash
   # Using PagerDuty
   curl -X POST "https://events.pagerduty.com/v2/enqueue" \
     -H "Content-Type: application/json" \
     -d '{
       "routing_key": "'$PAGERDUTY_ROUTING_KEY'",
       "event_action": "trigger",
       "payload": {
         "summary": "Critical LLM cost alert requires immediate attention",
         "severity": "critical",
         "source": "llm-cost-tracker"
       }
     }'
   ```

2. **Notify Finance Team** (for budget overruns)
   - Email: finance@terragonlabs.com
   - Include current spend, projected daily total, actions taken

3. **Security Team** (for suspected abuse)
   - Email: security@terragonlabs.com  
   - Include user patterns, IP addresses, request patterns

### Emergency Procedures

#### Complete Service Stop (Last Resort)
```bash
# Stop all LLM requests
kubectl scale deployment llm-cost-tracker --replicas=0 -n llm-cost-tracker

# Or enable maintenance mode
curl -X POST http://llm-cost-tracker:8000/api/v1/admin/maintenance \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "reason": "emergency_cost_control"}'
```

## Post-Incident Actions

1. **Document Actions Taken** - Record all steps and their effectiveness
2. **Update Thresholds** - Adjust alert thresholds based on incident learnings  
3. **Improve Automation** - Implement additional automated responses
4. **Team Communication** - Share lessons learned with engineering teams
5. **Runbook Updates** - Update this runbook based on new procedures

## Useful Queries

### Cost Analysis Queries
```sql
-- Hourly cost breakdown for investigation
SELECT 
  DATE_TRUNC('hour', timestamp) as hour,
  model,
  SUM(cost_usd) as cost,
  COUNT(*) as requests,
  AVG(total_tokens) as avg_tokens
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY hour, model
ORDER BY hour DESC, cost DESC;

-- Top users by cost (last 24h)
SELECT 
  user_id,
  application,
  SUM(cost_usd) as total_cost,
  COUNT(*) as requests,
  SUM(total_tokens) as total_tokens
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY user_id, application
ORDER BY total_cost DESC
LIMIT 20;

-- Cost efficiency by model
SELECT 
  model,
  COUNT(*) as requests,
  SUM(cost_usd) as total_cost,
  AVG(cost_usd) as avg_cost_per_request,
  SUM(total_tokens) as total_tokens,
  AVG(cost_usd * 1000 / total_tokens) as cost_per_1k_tokens
FROM cost_tracking.cost_records 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
  AND total_tokens > 0
GROUP BY model
ORDER BY cost_per_1k_tokens DESC;
```

This runbook provides comprehensive procedures for handling cost-related incidents while maintaining service availability and preventing budget overruns.