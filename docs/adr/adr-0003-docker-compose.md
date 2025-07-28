# ADR-0003: Docker Compose for Local Development

## Status
Accepted

## Context
Development teams need a consistent, reproducible local environment that includes the application, database, monitoring stack, and all dependencies without complex manual setup.

## Decision
We will use Docker Compose as the primary local development environment, providing pre-configured services for Postgres, Grafana, Prometheus, and the OTel Collector.

## Consequences

### Positive
- Consistent development environment across all team members
- Single command setup (`docker compose up -d`)
- Easy integration testing with real dependencies
- Port mappings allow local debugging while using containerized services
- Environment isolation prevents conflicts with system-installed services

### Negative
- Requires Docker installation and basic container knowledge
- Resource overhead from running multiple containers
- Potential networking complexity for some development scenarios

## Alternatives Considered
- **Local installations**: Inconsistent environments, complex setup documentation
- **Kubernetes (kind/minikube)**: Overkill complexity for local development
- **Vagrant**: Heavier resource usage, less portable across operating systems
- **Dev containers**: Good but requires VS Code, less tool-agnostic