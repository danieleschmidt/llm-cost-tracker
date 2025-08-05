# 🚀 Production Deployment Guide - Quantum Task Planner

This guide provides comprehensive instructions for deploying the Quantum Task Planner to production environments.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD with backup
- Network: 1 Gbps
- Load balancer (for high availability)

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- curl, jq, bc (for deployment scripts)
- SSL certificates (for HTTPS)

## 🔧 Pre-Deployment Setup

### 1. Clone and Prepare Repository

```bash
git clone https://github.com/your-org/quantum-task-planner.git
cd quantum-task-planner
```

### 2. Configure Environment

```bash
# Copy and customize environment configuration
cp .env.production.example .env.production

# Edit the configuration file
nano .env.production
```

**Critical Configuration Items:**

```bash
# Security - Generate strong passwords
POSTGRES_PASSWORD=your_very_secure_postgres_password_here
REDIS_PASSWORD=your_very_secure_redis_password_here
GRAFANA_ADMIN_PASSWORD=your_very_secure_grafana_password_here
JWT_SECRET_KEY=your_very_secure_jwt_secret_key_here_minimum_32_characters

# Domains - Update with your actual domains
API_DOMAIN=api.quantum-planner.your-domain.com
GRAFANA_DOMAIN=grafana.quantum-planner.your-domain.com

# Email for SSL certificates
ACME_EMAIL=admin@your-domain.com

# Compliance
DATA_CONTROLLER="Your Organization Name"
COMPLIANCE_REGIONS=eu_gdpr,us_ccpa  # Adjust based on your requirements
```

### 3. SSL/TLS Setup

For production, you need valid SSL certificates:

**Option A: Let's Encrypt (Recommended)**
```bash
# Traefik will automatically handle Let's Encrypt certificates
# Just ensure ACME_EMAIL is configured in .env.production
```

**Option B: Custom Certificates**
```bash
# Place your certificates in the appropriate location
mkdir -p /etc/ssl/certs /etc/ssl/private
cp your-cert.crt /etc/ssl/certs/quantum-planner.crt
cp your-key.key /etc/ssl/private/quantum-planner.key
chmod 600 /etc/ssl/private/quantum-planner.key
```

### 4. DNS Configuration

Configure DNS records for your domains:

```dns
api.quantum-planner.your-domain.com     A    YOUR_SERVER_IP
grafana.quantum-planner.your-domain.com A    YOUR_SERVER_IP
prometheus.quantum-planner.your-domain.com A YOUR_SERVER_IP
traefik.quantum-planner.your-domain.com A    YOUR_SERVER_IP
```

## 🚀 Deployment Process

### Automated Deployment (Recommended)

The automated deployment script handles the entire process:

```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Run full deployment
./scripts/deploy.sh deploy
```

The script will:
1. ✅ Check prerequisites
2. 💾 Create backups
3. 🧪 Run health checks
4. 🏗️ Build and test images
5. 🔄 Perform zero-downtime deployment
6. ✅ Verify deployment
7. 🧹 Clean up resources

### Manual Deployment

If you prefer manual control:

```bash
# 1. Build images
docker-compose -f docker-compose.production.yml build

# 2. Run quality gates
python scripts/quality_gates.py

# 3. Start services
docker-compose -f docker-compose.production.yml up -d

# 4. Verify health
curl -f http://localhost:8000/health
```

## 🔍 Post-Deployment Verification

### Health Checks

```bash
# Basic health check
curl -f http://localhost:8000/health

# Detailed health check  
curl -f http://localhost:8000/health/detailed

# Component-specific checks
curl -f http://localhost:8000/health/circuit-breakers
curl -f http://localhost:8000/metrics/cache
curl -f http://localhost:8000/metrics/concurrency
```

### Service Status

```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f quantum-task-planner
```

### Performance Verification

```bash
# Test API response time
time curl -s http://localhost:8000/health

# Check quantum planner functionality
curl -s http://localhost:8000/api/v1/quantum/system/state | jq
```

## 📊 Monitoring Setup

### Access Monitoring Dashboards

- **Grafana**: https://grafana.quantum-planner.your-domain.com
  - Username: `admin`
  - Password: `$GRAFANA_ADMIN_PASSWORD`

- **Prometheus**: https://prometheus.quantum-planner.your-domain.com

- **Traefik Dashboard**: https://traefik.quantum-planner.your-domain.com

### Key Metrics to Monitor

1. **Application Health**
   - Response time < 200ms
   - Error rate < 1%
   - Uptime > 99.9%

2. **Resource Utilization**
   - CPU < 70%
   - Memory < 80%
   - Disk < 85%

3. **Quantum Planner Specific**
   - Task success rate > 95%
   - Cache hit rate > 80%
   - Queue size < 100

## 🔧 Configuration Management

### Environment Variables

Update configurations without rebuilding:

```bash
# Edit environment
nano .env.production

# Restart services to apply changes
docker-compose -f docker-compose.production.yml restart quantum-task-planner
```

### Feature Toggles

Key features can be toggled via environment variables:

```bash
MONITORING_ENABLED=true          # Enable/disable monitoring
CACHE_ENABLED=true              # Enable/disable caching
LOAD_BALANCING_ENABLED=true     # Enable/disable load balancing
AUTO_SCALING_ENABLED=true       # Enable/disable auto-scaling
CIRCUIT_BREAKER_ENABLED=true    # Enable/disable circuit breakers
```

## 🛡️ Security Considerations

### Network Security

```bash
# Configure firewall (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Container Security

The deployment includes several security measures:
- Non-root user execution
- Resource limits
- Health checks
- Security headers
- Rate limiting

### Data Protection

- Database encryption at rest
- TLS encryption in transit
- PII anonymization
- GDPR compliance features
- Audit logging

## 📁 Backup and Recovery

### Automated Backups

Backups are created automatically during deployment:

```bash
# Manual backup
./scripts/deploy.sh backup

# View backup location
cat backups/latest_backup.txt
```

### Backup Contents

- Database dump
- Configuration files
- Docker images
- Application data

### Recovery Process

```bash
# Rollback to previous deployment
./scripts/deploy.sh rollback

# Restore from specific backup
# (See backup directory for available backups)
```

## 🚨 Troubleshooting

### Common Issues

**1. Service Won't Start**
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs service-name

# Check resource usage
docker stats

# Verify configuration
docker-compose -f docker-compose.production.yml config
```

**2. Database Connection Issues**
```bash
# Test database connectivity
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U quantum_user

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres
```

**3. SSL Certificate Issues**
```bash
# Check certificate status
openssl x509 -in /etc/ssl/certs/quantum-planner.crt -text -noout

# Verify Traefik configuration
docker-compose -f docker-compose.production.yml logs traefik
```

### Performance Issues

**1. High Response Times**
```bash
# Check system resources
htop
iostat -x 1

# Analyze slow queries
docker-compose -f docker-compose.production.yml exec postgres psql -U quantum_user -d quantum_db -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**2. Memory Issues**
```bash
# Check memory usage by container
docker stats --no-stream

# Adjust memory limits in docker-compose.production.yml
```

### Emergency Procedures

**1. Complete System Recovery**
```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Clean slate restart
docker system prune -a --volumes
./scripts/deploy.sh deploy
```

**2. Database Recovery**
```bash
# Restore from backup
backup_path=$(cat backups/latest_backup.txt)
docker-compose -f docker-compose.production.yml exec -T postgres psql -U quantum_user -d quantum_db < "$backup_path/database.sql"
```

## 📈 Scaling Considerations

### Horizontal Scaling

To scale beyond a single machine:

1. **Load Balancer Setup**
   - Use external load balancer (AWS ALB, GCP LB, etc.)
   - Configure health checks
   - Enable session affinity if needed

2. **Database Scaling**
   - Set up read replicas
   - Consider database sharding for very high loads
   - Use managed database services (AWS RDS, GCP Cloud SQL)

3. **Cache Scaling**
   - Use Redis Cluster
   - Implement cache partitioning
   - Consider managed Redis services

### Auto Scaling

The system includes built-in auto-scaling:
- Monitors queue sizes and resource usage
- Automatically adjusts worker counts
- Provides scaling metrics and history

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Update with zero downtime
git pull origin main
./scripts/deploy.sh deploy
```

### Maintenance Windows

For major updates requiring downtime:

1. Schedule maintenance window
2. Notify users
3. Create comprehensive backup
4. Perform update
5. Verify all functionality
6. Monitor for issues

### Regular Maintenance Tasks

- **Weekly**: Review logs and metrics
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance review and optimization
- **Annually**: Security audit and disaster recovery testing

## 📞 Support and Monitoring

### Log Locations

```bash
# Application logs
./logs/deployment.log
docker-compose -f docker-compose.production.yml logs

# System logs
/var/log/syslog
journalctl -u docker
```

### Monitoring Alerts

Configure alerts for:
- Service downtime
- High error rates
- Resource exhaustion
- Security events
- Performance degradation

### Contact Information

- **Emergency**: [Your emergency contact]
- **Technical Support**: [Your support team]
- **Security Issues**: [Your security team]

---

## 🎯 Production Checklist

Before going live, ensure:

- [ ] All environment variables configured
- [ ] SSL certificates valid and configured
- [ ] DNS records pointing to correct IP
- [ ] Firewall rules configured
- [ ] Backup system tested
- [ ] Monitoring dashboards accessible  
- [ ] Alert notifications working
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Rollback procedure tested
- [ ] Compliance requirements met
- [ ] Performance benchmarks established

---

**🎉 Congratulations! Your Quantum Task Planner is now running in production with enterprise-grade reliability, security, and scalability.**