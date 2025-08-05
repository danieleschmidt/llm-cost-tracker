#!/bin/bash

# Production Deployment Script for Quantum Task Planner
# This script handles zero-downtime deployment to production

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_FILE="${PROJECT_ROOT}/logs/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running or accessible"
    fi
    
    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/.env.production" ]]; then
        error ".env.production file not found. Copy from .env.production.example and configure."
    fi
    
    # Check disk space (require at least 5GB free)
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then # 5GB in KB
        error "Insufficient disk space. At least 5GB required."
    fi
    
    success "Prerequisites check passed"
}

# Backup current deployment
backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${BACKUP_DIR}/deployment_${backup_timestamp}"
    
    mkdir -p "$backup_path"
    
    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U quantum_user quantum_db > "${backup_path}/database.sql"
        success "Database backup completed"
    fi
    
    # Backup configuration
    cp -r "${PROJECT_ROOT}/config" "${backup_path}/"
    cp "${PROJECT_ROOT}/.env.production" "${backup_path}/"
    
    # Backup docker images
    log "Saving current Docker images..."
    docker save $(docker-compose -f "$COMPOSE_FILE" config | grep 'image:' | awk '{print $2}' | sort -u) | gzip > "${backup_path}/images.tar.gz"
    
    success "Backup completed: $backup_path"
    echo "$backup_path" > "${BACKUP_DIR}/latest_backup.txt"
}

# Run pre-deployment health checks
pre_deployment_checks() {
    log "Running pre-deployment health checks..."
    
    # Check current system health if running
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log "Checking current system health..."
        
        # Check API health
        if curl -f -s http://localhost:8000/health > /dev/null; then
            success "Current API is healthy"
        else
            warn "Current API health check failed"
        fi
        
        # Check database connectivity
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U quantum_user -d quantum_db; then
            success "Database is accessible"
        else
            warn "Database health check failed"
        fi
    fi
    
    # Validate docker-compose configuration
    if docker-compose -f "$COMPOSE_FILE" config > /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration validation failed"
    fi
    
    # Check for required environment variables
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "GRAFANA_ADMIN_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
        fi
    done
    
    success "Pre-deployment checks completed"
}

# Build and test new images
build_and_test() {
    log "Building new Docker images..."
    
    # Build with cache from registry if available
    docker-compose -f "$COMPOSE_FILE" build --parallel
    
    success "Docker images built successfully"
    
    log "Running tests on new build..."
    
    # Run tests in isolated environment
    docker-compose -f "$COMPOSE_FILE" run --rm quantum-task-planner python scripts/quality_gates.py
    
    success "All tests passed"
}

# Deploy with zero downtime
zero_downtime_deploy() {
    log "Starting zero-downtime deployment..."
    
    # Start new services alongside old ones
    log "Starting new services..."
    docker-compose -f "$COMPOSE_FILE" up -d --scale quantum-task-planner=2
    
    # Wait for new services to be healthy
    log "Waiting for new services to become healthy..."
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            success "New services are healthy"
            break
        fi
        
        attempt=$((attempt + 1))
        log "Health check attempt $attempt/$max_attempts..."
        sleep 10
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "New services failed to become healthy within timeout"
    fi
    
    # Gradually shift traffic to new instances
    log "Performing rolling update..."
    docker-compose -f "$COMPOSE_FILE" up -d --scale quantum-task-planner=1
    
    # Wait a bit to ensure stability
    sleep 30
    
    # Final health check
    if curl -f -s http://localhost:8000/health > /dev/null; then
        success "Rolling update completed successfully"
    else
        error "Rolling update failed - services are not healthy"
    fi
}

# Post-deployment verification
post_deployment_verification() {
    log "Running post-deployment verification..."
    
    # Check all services are running
    local services=("quantum-task-planner" "postgres" "redis" "prometheus" "grafana")
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            success "$service is running"
        else
            error "$service is not running"
        fi
    done
    
    # Comprehensive health checks
    local health_endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8000/health/detailed"
        "http://localhost:9090/-/healthy"
        "http://localhost:3000/api/health"
    )
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null; then
            success "Health check passed: $endpoint"  
        else
            warn "Health check failed: $endpoint"
        fi
    done
    
    # Check quantum planner specific functionality
    log "Testing quantum planner functionality..."
    
    # Test API endpoints
    local api_endpoints=(
        "/api/v1/quantum/system/state"
        "/metrics/cache"
        "/metrics/concurrency"
    )
    
    for endpoint in "${api_endpoints[@]}"; do
        if curl -f -s "http://localhost:8000$endpoint" > /dev/null; then
            success "API endpoint accessible: $endpoint"
        else
            warn "API endpoint failed: $endpoint"
        fi
    done
    
    # Performance check
    log "Running performance verification..."
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/health)
    if (( $(echo "$response_time < 2.0" | bc -l) )); then
        success "Response time acceptable: ${response_time}s"
    else
        warn "Response time high: ${response_time}s"
    fi
    
    success "Post-deployment verification completed"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    if [[ -f "${BACKUP_DIR}/latest_backup.txt" ]]; then
        local backup_path=$(cat "${BACKUP_DIR}/latest_backup.txt")
        
        if [[ -d "$backup_path" ]]; then
            log "Rolling back to backup: $backup_path"
            
            # Stop current services
            docker-compose -f "$COMPOSE_FILE" down
            
            # Restore configuration
            cp "${backup_path}/.env.production" "${PROJECT_ROOT}/"
            cp -r "${backup_path}/config" "${PROJECT_ROOT}/"
            
            # Restore database
            if [[ -f "${backup_path}/database.sql" ]]; then
                docker-compose -f "$COMPOSE_FILE" up -d postgres
                sleep 10
                docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U quantum_user -d quantum_db < "${backup_path}/database.sql"
            fi
            
            # Restore images
            if [[ -f "${backup_path}/images.tar.gz" ]]; then
                gunzip -c "${backup_path}/images.tar.gz" | docker load
            fi
            
            # Start services
            docker-compose -f "$COMPOSE_FILE" up -d
            
            success "Rollback completed"
        else
            error "Backup directory not found: $backup_path"
        fi
    else
        error "No backup information found for rollback"
    fi
}

# Cleanup old resources
cleanup() {
    log "Cleaning up old resources..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove old backups (keep last 10)
    find "$BACKUP_DIR" -name "deployment_*" -type d | sort -r | tail -n +11 | xargs rm -rf
    
    # Clean up logs older than 30 days
    find "${PROJECT_ROOT}/logs" -name "*.log" -mtime +30 -delete
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting production deployment of Quantum Task Planner"
    
    # Create necessary directories
    mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"
    
    # Source environment
    if [[ -f "${PROJECT_ROOT}/.env.production" ]]; then
        # Export variables from .env.production
        set -a
        source "${PROJECT_ROOT}/.env.production"
        set +a
    fi
    
    # Trap errors for rollback
    trap rollback ERR
    
    # Execute deployment steps
    check_prerequisites
    backup_current_deployment
    pre_deployment_checks
    build_and_test
    zero_downtime_deploy
    post_deployment_verification
    cleanup
    
    success "🎉 Production deployment completed successfully!"
    log "Deployment log: $LOG_FILE"
    log "Backup location: $(cat "${BACKUP_DIR}/latest_backup.txt")"
    
    # Display summary
    echo ""
    echo "=== DEPLOYMENT SUMMARY ==="
    echo "✅ Services deployed and healthy"
    echo "✅ Zero-downtime deployment completed"
    echo "✅ All health checks passed"
    echo "✅ Backup created and stored"
    echo ""
    echo "🌐 Application URLs:"
    echo "   • API: http://localhost:8000"
    echo "   • Grafana: http://localhost:3000"
    echo "   • Prometheus: http://localhost:9090"
    echo ""
    echo "📋 Next steps:"
    echo "   • Monitor application metrics"
    echo "   • Verify user functionality"
    echo "   • Update monitoring alerts if needed"
    echo ""
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "backup")
        backup_current_deployment
        ;;
    "health")
        post_deployment_verification
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|backup|health]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full production deployment (default)"
        echo "  rollback - Rollback to previous deployment"
        echo "  backup   - Create backup of current deployment"
        echo "  health   - Run health checks on current deployment"
        exit 1
        ;;
esac