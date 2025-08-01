# Deployment Guide for LLM Cost Tracker

This guide covers deployment strategies, containerization, and infrastructure setup for the LLM Cost Tracker.

## Overview

The LLM Cost Tracker supports multiple deployment strategies:

1. **Docker Compose** - Local development and small-scale deployments
2. **Docker Swarm** - Multi-node container orchestration
3. **Kubernetes** - Enterprise-grade orchestration (Helm charts available)
4. **Cloud Platforms** - AWS ECS, GCP Cloud Run, Azure Container Instances

## Container Architecture

### Multi-Stage Docker Build

Our Dockerfile uses a multi-stage build approach:

```dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Stage 2: Development - Full development environment
FROM builder as development

# Stage 3: Production - Optimized runtime
FROM python:3.11-slim as production

# Stage 4: Security - Vulnerability scanning
FROM production as security
```

### Build Targets

```bash
# Development build (includes dev dependencies)
docker build --target development -t llm-cost-tracker:dev .

# Production build (optimized)
docker build --target production -t llm-cost-tracker:prod .

# Security scanning build
docker build --target security -t llm-cost-tracker:security .
```

## Local Development

### Quick Start

```bash
# Clone and setup
git clone https://github.com/terragon-labs/llm-cost-tracker
cd llm-cost-tracker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f llm-cost-tracker
```

### Development Workflow

```bash
# Start services with hot reload
make docker-up
make dev

# Run tests in container
docker-compose exec llm-cost-tracker pytest

# Access database
docker-compose exec postgres psql -U postgres -d llm_metrics

# View metrics
curl http://localhost:9090  # Prometheus
curl http://localhost:3000  # Grafana
```

### Environment Configuration

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_metrics

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=llm-cost-tracker

# API Keys (for production)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key_here

# Features
ENABLE_BUDGET_ALERTS=true
ENABLE_MODEL_SWAPPING=true
ENABLE_METRICS_EXPORT=true

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

## Production Deployment

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB
- Network: 100Mbps

**Recommended Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 100GB SSD
- Network: 1Gbps

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  llm-cost-tracker:
    image: llm-cost-tracker:latest
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/llm_metrics
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Security Configuration

```bash
# Create dedicated user
sudo useradd -r -s /bin/false llm-cost-tracker

# Set up directories with proper permissions
sudo mkdir -p /opt/llm-cost-tracker/{data,logs,config}
sudo chown -R llm-cost-tracker:llm-cost-tracker /opt/llm-cost-tracker
sudo chmod 750 /opt/llm-cost-tracker

# Configure firewall
sudo ufw allow 8000/tcp  # Application
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus
```

## Kubernetes Deployment

### Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-cost-tracker
  labels:
    name: llm-cost-tracker
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-cost-tracker-config
  namespace: llm-cost-tracker
data:
  DATABASE_URL: "postgresql://user:pass@postgres:5432/llm_metrics"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
  LOG_LEVEL: "INFO"
  ENABLE_BUDGET_ALERTS: "true"
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-cost-tracker
  namespace: llm-cost-tracker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-cost-tracker
  template:
    metadata:
      labels:
        app: llm-cost-tracker
    spec:
      containers:
      - name: llm-cost-tracker
        image: llm-cost-tracker:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: llm-cost-tracker-config
        - secretRef:
            name: llm-cost-tracker-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-cost-tracker-service
  namespace: llm-cost-tracker
spec:
  selector:
    app: llm-cost-tracker
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Helm Chart

```bash
# Install with Helm
helm repo add llm-cost-tracker https://charts.terragon-labs.com
helm install llm-cost-tracker llm-cost-tracker/llm-cost-tracker \
  --namespace llm-cost-tracker \
  --create-namespace \
  --values values.yaml
```

## Cloud Platform Deployments

### AWS ECS

```json
{
  "family": "llm-cost-tracker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "llm-cost-tracker",
      "image": "your-account.dkr.ecr.region.amazonaws.com/llm-cost-tracker:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/llm_metrics"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-cost-tracker",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: llm-cost-tracker
  namespace: default
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "1Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/project-id/llm-cost-tracker:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@sql-proxy:5432/llm_metrics"
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Azure Container Instances

```bash
az container create \
  --resource-group llm-cost-tracker-rg \
  --name llm-cost-tracker \
  --image llm-cost-tracker:latest \
  --dns-name-label llm-cost-tracker \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL='postgresql://user:pass@postgres:5432/llm_metrics' \
    LOG_LEVEL='INFO' \
  --cpu 2 \
  --memory 4
```

## Database Setup

### PostgreSQL Production Setup

```sql
-- Create database and user
CREATE DATABASE llm_metrics;
CREATE USER llm_cost_tracker WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE llm_metrics TO llm_cost_tracker;

-- Connect to database
\c llm_metrics

-- Create schema
CREATE SCHEMA IF NOT EXISTS cost_tracking;
GRANT ALL ON SCHEMA cost_tracking TO llm_cost_tracker;

-- Performance optimizations
CREATE INDEX CONCURRENTLY idx_cost_records_timestamp 
ON cost_tracking.cost_records(timestamp);

CREATE INDEX CONCURRENTLY idx_cost_records_user_id 
ON cost_tracking.cost_records(user_id);

CREATE INDEX CONCURRENTLY idx_cost_records_model 
ON cost_tracking.cost_records(model);
```

### Database Migration

```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Add new table"

# Rollback migration
alembic downgrade -1
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert-rules.yml"

scrape_configs:
  - job_name: 'llm-cost-tracker'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "LLM Cost Tracker - Production",
    "panels": [
      {
        "title": "Cost per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_cost_total[1h])",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])",
            "legendFormat": "{{method}}"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alert-rules.yml
groups:
  - name: llm-cost-tracker
    rules:
      - alert: HighCostRate
        expr: rate(llm_cost_total[1h]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM cost rate detected"
          description: "Cost rate is {{ $value }} per hour"

      - alert: ServiceDown
        expr: up{job="llm-cost-tracker"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM Cost Tracker service is down"
```

## Load Balancing

### Nginx Configuration

```nginx
upstream llm_cost_tracker {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name llm-cost-tracker.example.com;

    location / {
        proxy_pass http://llm_cost_tracker;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://llm_cost_tracker;
    }
}
```

### HAProxy Configuration

```
backend llm_cost_tracker
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server app1 app1:8000 check
    server app2 app2:8000 check
    server app3 app3:8000 check

frontend llm_cost_tracker_frontend
    bind *:80
    default_backend llm_cost_tracker
```

## Security Hardening

### SSL/TLS Configuration

```bash
# Generate certificates with Let's Encrypt
certbot --nginx -d llm-cost-tracker.example.com

# Or use custom certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/llm-cost-tracker.key \
    -out /etc/ssl/certs/llm-cost-tracker.crt
```

### Environment Security

```bash
# Use secrets management
export DATABASE_URL=$(aws secretsmanager get-secret-value \
    --secret-id prod/llm-cost-tracker/db-url \
    --query SecretString --output text)

# Or use Kubernetes secrets
kubectl create secret generic llm-cost-tracker-secrets \
    --from-literal=database-url='postgresql://...' \
    --from-literal=secret-key='...'
```

### Network Security

```bash
# Docker network isolation
docker network create --driver bridge llm-cost-tracker-network

# Kubernetes network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-cost-tracker-netpol
spec:
  podSelector:
    matchLabels:
      app: llm-cost-tracker
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/llm-cost-tracker"
DB_NAME="llm_metrics"

pg_dump -h postgres -U postgres -d $DB_NAME \
    --no-password --clean --create \
    | gzip > "$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### Configuration Backup

```bash
# Backup configuration files
tar -czf config-backup-$(date +%Y%m%d).tar.gz \
    config/ \
    .env \
    docker-compose.yml
```

### Disaster Recovery

```bash
# Restore from backup
gunzip -c backup_20240101_120000.sql.gz | \
    psql -h postgres -U postgres -d llm_metrics

# Verify data integrity
psql -h postgres -U postgres -d llm_metrics \
    -c "SELECT COUNT(*) FROM cost_tracking.cost_records;"
```

## Performance Optimization

### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### Application Tuning

```python
# uvicorn production settings
uvicorn llm_cost_tracker.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**
   ```bash
   # Check logs
   docker logs llm-cost-tracker
   
   # Verify environment variables
   docker exec llm-cost-tracker env
   ```

2. **Database Connection Issues**
   ```bash
   # Test connection
   docker exec llm-cost-tracker \
       python -c "from llm_cost_tracker.database import test_connection; test_connection()"
   ```

3. **High Memory Usage**
   ```bash
   # Monitor resource usage
   docker stats llm-cost-tracker
   
   # Analyze memory patterns
   docker exec llm-cost-tracker python -m memory_profiler main.py
   ```

### Health Checks

```bash
# Application health
curl -f http://localhost:8000/health

# Database health
curl -f http://localhost:8000/health/db

# Dependencies health
curl -f http://localhost:8000/health/dependencies
```

### Log Analysis

```bash
# Structured logging with jq
docker logs llm-cost-tracker | jq '.level == "ERROR"'

# Performance metrics
docker logs llm-cost-tracker | grep "response_time" | \
    awk '{print $5}' | sort -n | tail -10
```

This deployment guide provides comprehensive coverage for all deployment scenarios from local development to enterprise production environments.