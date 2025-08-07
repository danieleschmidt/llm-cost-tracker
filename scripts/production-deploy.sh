#!/bin/bash

# Production Deployment Script for Sentiment Analyzer Pro
# This script handles zero-downtime deployment with health checks and rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
VERSION="${VERSION:-$(date +%Y%m%d-%H%M%S)}"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_ENABLED="${ROLLBACK_ENABLED:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
error_handler() {
    local line_number=$1
    log_error "Error on line $line_number. Deployment failed!"
    
    if [[ "$ROLLBACK_ENABLED" == "true" ]]; then
        log_warning "Attempting automatic rollback..."
        rollback_deployment
    fi
    
    exit 1
}

trap 'error_handler $LINENO' ERR

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$PROJECT_ROOT/.env.production" ]]; then
        log_error "Production environment file (.env.production) not found"
        log_info "Copy .env.production.example to .env.production and configure it"
        exit 1
    fi
    
    # Check required environment variables
    source "$PROJECT_ROOT/.env.production"
    
    required_vars=("POSTGRES_PASSWORD" "SECRET_KEY" "JWT_SECRET_KEY" "OPENAI_API_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check system resources
    available_memory=$(free -g | awk 'NR==2{printf "%.1f", $7}')
    if (( $(echo "$available_memory < 4.0" | bc -l) )); then
        log_warning "Low available memory: ${available_memory}GB. Consider upgrading for better performance."
    fi
    
    # Check disk space
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if (( available_space < 10 )); then
        log_error "Insufficient disk space: ${available_space}GB available. Need at least 10GB."
        exit 1
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build and tag images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build -f Dockerfile.production \
        --target production \
        --tag "sentiment-analyzer-pro:$VERSION" \
        --tag "sentiment-analyzer-pro:latest" \
        .
    
    # Verify image build
    if ! docker images | grep -q "sentiment-analyzer-pro.*$VERSION"; then
        log_error "Image build failed"
        exit 1
    fi
    
    log_success "Docker images built successfully"
}

# Database migration
run_migrations() {
    log_info "Running database migrations..."
    
    # Start database if not running
    docker-compose -f docker-compose.production.yml up -d postgres redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    max_attempts=30
    attempt=1
    
    while ! docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U sentiment_user -d sentiment_analyzer_prod; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Database failed to start after $max_attempts attempts"
            exit 1
        fi
        
        log_info "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # Run migrations (placeholder - would use actual migration tool)
    log_info "Running database schema migrations..."
    docker-compose -f docker-compose.production.yml exec -T postgres psql -U sentiment_user -d sentiment_analyzer_prod -c "
        CREATE TABLE IF NOT EXISTS sentiment_analysis_requests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_id VARCHAR(64) UNIQUE NOT NULL,
            user_id VARCHAR(128),
            text_hash VARCHAR(128) NOT NULL,
            text_length INTEGER NOT NULL,
            language VARCHAR(5) DEFAULT 'en',
            sentiment_label VARCHAR(20),
            confidence_score FLOAT,
            processing_time_ms FLOAT,
            model_used VARCHAR(64),
            cost_usd FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE
        );
        
        CREATE INDEX IF NOT EXISTS idx_sentiment_created_at ON sentiment_analysis_requests(created_at);
        CREATE INDEX IF NOT EXISTS idx_sentiment_user_id ON sentiment_analysis_requests(user_id);
    "
    
    log_success "Database migrations completed"
}

# Deploy services with zero-downtime
deploy_services() {
    log_info "Deploying services with zero-downtime strategy..."
    
    cd "$PROJECT_ROOT"
    
    # Create backup of current state
    if [[ "$ROLLBACK_ENABLED" == "true" ]]; then
        log_info "Creating deployment backup..."
        mkdir -p "deployments/$VERSION"
        cp -r config/ "deployments/$VERSION/" || true
        docker-compose -f docker-compose.production.yml config > "deployments/$VERSION/docker-compose.backup.yml"
    fi
    
    # Deploy infrastructure services first
    log_info "Starting infrastructure services..."
    docker-compose -f docker-compose.production.yml up -d \
        postgres \
        redis \
        prometheus \
        grafana \
        otel-collector
    
    # Wait for infrastructure to be ready
    wait_for_service "postgres" "pg_isready -U sentiment_user -d sentiment_analyzer_prod"
    wait_for_service "redis" "redis-cli ping"
    wait_for_service "prometheus" "curl -f http://localhost:9090/-/healthy"
    
    # Deploy main application
    log_info "Deploying main application..."
    docker-compose -f docker-compose.production.yml up -d sentiment-analyzer
    
    # Deploy reverse proxy last
    log_info "Deploying reverse proxy..."
    docker-compose -f docker-compose.production.yml up -d traefik
    
    log_success "All services deployed"
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local health_check_cmd=$2
    local max_attempts=60
    local attempt=1
    
    log_info "Waiting for $service_name to be ready..."
    
    while ! docker-compose -f docker-compose.production.yml exec -T "$service_name" $health_check_cmd &>/dev/null; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "$service_name failed to start after $max_attempts attempts"
            exit 1
        fi
        
        log_info "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    log_success "$service_name is ready"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    local health_url="http://localhost:8000/health"
    local max_attempts=60
    local attempt=1
    
    while ! curl -f "$health_url" &>/dev/null; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Health check... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    # Extended health check
    log_info "Running extended health checks..."
    
    # Check API endpoints
    if ! curl -f "http://localhost:8000/api/v1/sentiment/health" &>/dev/null; then
        log_error "Sentiment analysis health check failed"
        return 1
    fi
    
    # Check database connectivity
    if ! docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U sentiment_user -d sentiment_analyzer_prod &>/dev/null; then
        log_error "Database health check failed"
        return 1
    fi
    
    # Check cache connectivity
    if ! docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping | grep -q "PONG"; then
        log_error "Redis health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
    return 0
}

# Performance benchmark
performance_check() {
    log_info "Running performance benchmarks..."
    
    # Simple performance test
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message for performance benchmarking."}' \
        http://localhost:8000/api/v1/sentiment/analyze)
    
    log_info "API response time: ${response_time}s"
    
    # Check if response time is acceptable
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        log_warning "High API response time: ${response_time}s (expected < 2.0s)"
    else
        log_success "API response time within acceptable limits"
    fi
}

# Security check
security_check() {
    log_info "Running security checks..."
    
    # Check for exposed secrets
    log_info "Checking for exposed environment variables..."
    if curl -s "http://localhost:8000/config" | grep -i "password\|secret\|key" &>/dev/null; then
        log_error "Potential secret exposure detected"
        return 1
    fi
    
    # Check security headers
    log_info "Checking security headers..."
    headers=$(curl -I -s http://localhost:8000/health)
    
    if ! echo "$headers" | grep -q "X-Content-Type-Options"; then
        log_warning "X-Content-Type-Options header missing"
    fi
    
    if ! echo "$headers" | grep -q "X-Frame-Options"; then
        log_warning "X-Frame-Options header missing"
    fi
    
    log_success "Security checks completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Find latest backup
    local latest_backup
    latest_backup=$(ls -t deployments/ | head -n 2 | tail -n 1 2>/dev/null || echo "")
    
    if [[ -z "$latest_backup" ]]; then
        log_error "No backup found for rollback"
        return 1
    fi
    
    log_info "Rolling back to version: $latest_backup"
    
    # Stop current services
    docker-compose -f docker-compose.production.yml down
    
    # Restore backup configuration
    if [[ -f "deployments/$latest_backup/docker-compose.backup.yml" ]]; then
        cp "deployments/$latest_backup/docker-compose.backup.yml" docker-compose.production.yml
        docker-compose -f docker-compose.production.yml up -d
        
        if health_check; then
            log_success "Rollback completed successfully"
        else
            log_error "Rollback health check failed"
            return 1
        fi
    else
        log_error "Backup configuration not found"
        return 1
    fi
}

# Cleanup old deployments
cleanup_old_deployments() {
    log_info "Cleaning up old deployments..."
    
    # Keep last 5 deployments
    if [[ -d "deployments" ]]; then
        local old_deployments
        old_deployments=$(ls -t deployments/ | tail -n +6)
        
        for deployment in $old_deployments; do
            log_info "Removing old deployment: $deployment"
            rm -rf "deployments/$deployment"
        done
    fi
    
    # Clean up old Docker images
    log_info "Cleaning up old Docker images..."
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting Sentiment Analyzer Pro deployment (version: $VERSION)"
    log_info "Environment: $DEPLOY_ENV"
    log_info "Rollback enabled: $ROLLBACK_ENABLED"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run deployment steps
    pre_deployment_checks
    build_images
    run_migrations
    deploy_services
    
    # Health and performance checks
    if health_check; then
        log_success "Deployment health checks passed"
        performance_check
        security_check
        cleanup_old_deployments
        
        log_success "🚀 Sentiment Analyzer Pro deployment completed successfully!"
        log_info "Application is available at: http://localhost:8000"
        log_info "API documentation: http://localhost:8000/docs"
        log_info "Grafana dashboard: http://localhost:3000"
        log_info "Prometheus metrics: http://localhost:9090"
        
    else
        log_error "Deployment health checks failed"
        exit 1
    fi
    
    # Display deployment summary
    echo
    log_success "=== DEPLOYMENT SUMMARY ==="
    log_info "Version: $VERSION"
    log_info "Environment: $DEPLOY_ENV"
    log_info "Services deployed: sentiment-analyzer, postgres, redis, prometheus, grafana, traefik"
    log_info "Health checks: ✅ PASSED"
    log_info "Performance checks: ✅ COMPLETED"
    log_info "Security checks: ✅ COMPLETED"
    echo
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "health-check")
        health_check
        ;;
    "rollback")
        rollback_deployment
        ;;
    "cleanup")
        cleanup_old_deployments
        ;;
    *)
        echo "Usage: $0 {deploy|health-check|rollback|cleanup}"
        echo
        echo "Commands:"
        echo "  deploy      - Full deployment with health checks"
        echo "  health-check - Run health checks only"
        echo "  rollback    - Rollback to previous deployment"
        echo "  cleanup     - Clean up old deployments and images"
        exit 1
        ;;
esac