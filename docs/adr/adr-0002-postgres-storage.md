# ADR-0002: Postgres for Data Storage

## Status
Accepted

## Context
The system requires persistent storage for LLM metrics, budget rules, and historical cost data with support for complex queries, aggregations, and real-time analytics.

## Decision
We will use PostgreSQL as the primary data storage backend with asyncpg for high-performance async database operations.

## Consequences

### Positive
- ACID compliance ensures data consistency for budget calculations
- Rich SQL feature set supports complex cost analytics queries
- Excellent performance for time-series data with proper indexing
- Strong ecosystem support and operational tooling
- Native JSON support for flexible schema evolution

### Negative
- Additional operational complexity compared to file-based storage
- Requires database management and backup strategies
- Single point of failure without proper clustering

## Alternatives Considered
- **ClickHouse**: Better for pure analytics but lacks ACID guarantees needed for budget enforcement
- **BigQuery**: Cloud-only solution, vendor lock-in concerns
- **InfluxDB**: Specialized for time-series but less flexible for relational budget rules
- **SQLite**: Simpler but inadequate for concurrent access patterns