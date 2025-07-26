# Security Policy

## Overview

LLM Cost Tracker handles sensitive API keys and cost data. This document outlines security best practices and our approach to maintaining a secure system.

## Security Best Practices

### 1. API Key Management

**NEVER hardcode API keys in source code.**

✅ **Do:**
```bash
# Set via environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

❌ **Don't:**
```python
# NEVER do this
api_key = "sk-1234567890abcdef"
```

### 2. Environment Configuration

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Set secure values in `.env`:**
   - Use strong, unique API keys
   - Enable HTTPS in production (`https://` URLs only)
   - Set `DEBUG=false` in production
   - Configure appropriate `ALLOWED_ORIGINS`

3. **Database Security:**
   - Use strong PostgreSQL passwords
   - Enable SSL/TLS for database connections in production
   - Regularly rotate database credentials

### 3. Docker Security

- Run containers as non-root users
- Use specific image tags, not `latest`
- Regularly update base images for security patches
- Limit container resource usage

### 4. API Security

The application implements several security measures:

- **Rate limiting**: Prevents API abuse
- **Request size limits**: Prevents DoS attacks
- **Input sanitization**: Prevents injection attacks
- **Security headers**: Protects against common web vulnerabilities
- **API key authentication**: For sensitive endpoints

### 5. Data Protection

- **Encryption at rest**: Use encrypted storage for sensitive data
- **Encryption in transit**: All API calls use HTTPS
- **Data masking**: Sensitive data is automatically redacted in logs
- **Access control**: Implement proper authentication and authorization

## Production Deployment Security

### Environment Variables

Required security configurations for production:

```bash
# Database (use SSL)
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# API Keys (from secure vault)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VELLUM_API_KEY=vellum-...

# Security
DEBUG=false
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://yourdomain.com

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Docker Compose Security

For production, modify `docker-compose.yml`:

1. **Use external networks**
2. **Enable TLS/SSL**
3. **Set resource limits**
4. **Use secrets management**

Example production overrides:
```yaml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password

secrets:
  postgres_password:
    external: true
```

### Monitoring and Alerting

- **Enable security monitoring**: Monitor for unusual API usage patterns
- **Set up alerts**: Configure alerts for cost thresholds and security events
- **Regular audits**: Review access logs and API usage patterns
- **Incident response**: Have a plan for security incidents

## Vulnerability Reporting

If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. Email security findings to: security@terragonlabs.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested mitigation (if any)

We will acknowledge receipt within 24 hours and provide a timeline for resolution.

## Security Updates

- Monitor dependencies for security updates
- Subscribe to security advisories for:
  - OpenTelemetry
  - FastAPI
  - PostgreSQL
  - Docker base images

## Compliance Considerations

When deploying LLM Cost Tracker:

- **Data residency**: Ensure data is stored in appropriate regions
- **Privacy**: Implement appropriate data retention policies  
- **Audit trails**: Maintain logs for compliance requirements
- **Access controls**: Implement proper user authentication and authorization

## Security Checklist

Before deploying to production:

- [ ] All API keys set via environment variables
- [ ] `.env` file excluded from version control
- [ ] Database uses SSL/TLS connections
- [ ] HTTPS enabled for all external communications
- [ ] DEBUG mode disabled
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Security scanning completed
- [ ] Access controls implemented
- [ ] Incident response plan documented

## Resources

- [OWASP Security Guidelines](https://owasp.org/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)
- [Docker Security](https://docs.docker.com/engine/security/)

---

**Remember: Security is everyone's responsibility. When in doubt, choose the more secure option.**