# ADR-0001: OpenTelemetry for Metrics Collection

## Status
Accepted

## Context
The system needs to collect and transmit LLM usage metrics (tokens, costs, latency) in a standardized, vendor-neutral format that can integrate with existing observability infrastructure.

## Decision
We will use OpenTelemetry (OTel) as the primary metrics collection and transmission framework, specifically leveraging the OTLP (OpenTelemetry Protocol) for data export.

## Consequences

### Positive
- Industry-standard observability framework with wide ecosystem support
- Vendor-neutral approach allows flexibility in backend storage
- Rich instrumentation libraries for Python and LangChain integration
- Native Prometheus metrics export capability
- Future-proof solution as OTel becomes the observability standard

### Negative
- Additional complexity compared to direct metrics collection
- Learning curve for team members unfamiliar with OTel concepts
- Potential performance overhead from instrumentation

## Alternatives Considered
- **Direct database writes**: Simpler but less flexible and harder to integrate with existing observability
- **Prometheus client library**: Limited to Prometheus ecosystem, less standardized
- **Custom metrics format**: Vendor lock-in and maintenance burden