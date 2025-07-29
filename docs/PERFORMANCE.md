# Performance Guide

This document provides guidelines for optimizing and monitoring the performance of LLM Cost Tracker.

## Performance Benchmarks

### Target Performance Metrics

| Metric | Target | Critical Threshold |
|--------|--------|--------------------|
| API Response Time (p95) | < 200ms | < 500ms |
| Trace Ingestion Rate | > 1000 traces/sec | > 500 traces/sec |
| Database Query Time (p95) | < 100ms | < 300ms |
| Memory Usage | < 2GB | < 4GB |
| CPU Usage (avg) | < 70% | < 90% |

### Load Testing Results

Performance characteristics under various load conditions:

```
Concurrent Users: 100
Request Rate: 500 req/sec
Duration: 10 minutes

Results:
- Average Response Time: 89ms
- 95th Percentile: 156ms
- 99th Percentile: 234ms
- Error Rate: 0.02%
- Throughput: 495 req/sec
```

## Optimization Strategies

### Database Optimization

1. **Index Strategy**
   ```sql
   -- Core indexes for performance
   CREATE INDEX CONCURRENTLY idx_traces_timestamp ON traces (timestamp DESC);
   CREATE INDEX CONCURRENTLY idx_traces_application ON traces (application, timestamp);
   CREATE INDEX CONCURRENTLY idx_traces_composite ON traces (application, model, timestamp);
   
   -- Partial indexes for common queries
   CREATE INDEX CONCURRENTLY idx_traces_recent 
   ON traces (timestamp) 
   WHERE timestamp > NOW() - INTERVAL '7 days';
   ```

2. **Query Optimization**
   ```python
   # Use efficient aggregation queries
   SELECT 
       application,
       model,
       DATE_TRUNC('hour', timestamp) as hour,
       SUM(cost_usd) as total_cost,
       SUM(total_tokens) as total_tokens,
       COUNT(*) as request_count
   FROM traces 
   WHERE timestamp >= $1 AND timestamp < $2
   GROUP BY application, model, hour
   ORDER BY hour DESC;
   ```

3. **Connection Pooling**
   ```python
   # Optimal connection pool settings
   DATABASE_CONFIG = {
       "min_connections": 5,
       "max_connections": 20,
       "max_idle_time": 300,
       "retry_attempts": 3,
       "statement_timeout": 30000
   }
   ```

### Application Optimization

1. **Async Processing**
   ```python
   # Use async/await for database operations
   async def batch_insert_traces(traces: List[TraceData]):
       async with get_db_connection() as conn:
           await conn.executemany(INSERT_TRACE_SQL, traces)
   ```

2. **Caching Strategy**
   ```python
   # Redis caching for frequent queries
   @cache(expire=300)  # 5-minute cache
   async def get_application_summary(app_name: str, time_range: str):
       return await db.get_app_metrics(app_name, time_range)
   ```

3. **Request Batching**
   ```python
   # Batch multiple traces in single request
   async def process_trace_batch(traces: List[TraceData]):
       if len(traces) >= BATCH_SIZE:
           await flush_batch(traces)
   ```

### Memory Management

1. **Efficient Data Structures**
   ```python
   # Use dataclasses with slots for memory efficiency
   @dataclass(slots=True)
   class TraceData:
       trace_id: str
       timestamp: int
       cost_usd: float
       total_tokens: int
   ```

2. **Streaming Responses**
   ```python
   # Stream large datasets instead of loading all in memory
   async def stream_cost_report(query_params):
       async for batch in db.stream_query(query_params):
           yield batch
   ```

### Monitoring Performance

1. **Application Metrics**
   ```python
   # Custom Prometheus metrics
   from prometheus_client import Counter, Histogram, Gauge
   
   REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
   REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
   ACTIVE_CONNECTIONS = Gauge('db_connections_active', 'Active database connections')
   ```

2. **Database Monitoring**
   ```sql
   -- Monitor slow queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   WHERE mean_time > 100
   ORDER BY mean_time DESC;
   
   -- Monitor index usage
   SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   ORDER BY idx_tup_read DESC;
   ```

## Load Testing

### Running Performance Tests

1. **Local Testing**
   ```bash
   # Start the stack
   docker-compose up -d
   
   # Install locust
   pip install locust
   
   # Run load test
   locust -f locustfile.py --host=http://localhost:8000
   ```

2. **Production-like Testing**
   ```bash
   # High-load scenario
   locust -f locustfile.py \
     --host=https://your-api.com \
     --users 500 \
     --spawn-rate 10 \
     --run-time 10m \
     --html=load_test_report.html
   ```

### Load Test Scenarios

1. **Normal Operations**
   - 100 concurrent users
   - 10 requests/second/user
   - Mix of read/write operations (70/30)

2. **Peak Load**
   - 500 concurrent users
   - 15 requests/second/user
   - High trace ingestion volume

3. **Stress Testing**
   - 1000+ concurrent users
   - Maximum sustainable load
   - Identify breaking points

## Performance Troubleshooting

### Common Issues

1. **High Database CPU**
   ```sql
   -- Find expensive queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY total_time DESC
   LIMIT 10;
   ```
   
   **Solutions:**
   - Add missing indexes
   - Optimize query patterns
   - Increase connection pool size

2. **Memory Leaks**
   ```bash
   # Monitor memory usage
   docker stats llm-cost-tracker
   
   # Profile memory usage
   py-spy top --pid <process_id>
   ```
   
   **Solutions:**
   - Review async task management
   - Check for unclosed connections
   - Implement proper garbage collection

3. **Slow API Responses**
   ```python
   # Add request timing middleware
   @app.middleware("http")
   async def add_process_time_header(request: Request, call_next):
       start_time = time.time()
       response = await call_next(request)
       process_time = time.time() - start_time
       response.headers["X-Process-Time"] = str(process_time)
       return response
   ```

### Performance Monitoring Dashboard

Key metrics to monitor in Grafana:

1. **Application Metrics**
   - Request rate and response times
   - Error rates and status codes
   - Memory and CPU usage
   - Queue depths and processing times

2. **Database Performance**
   - Connection pool utilization
   - Query execution times
   - Lock waits and deadlocks
   - Index hit ratios

3. **System Resources**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network bandwidth

## Capacity Planning

### Scaling Guidelines

1. **Vertical Scaling**
   - Add more CPU cores for compute-intensive operations
   - Increase memory for caching and buffering
   - Use faster storage (SSD) for database workloads

2. **Horizontal Scaling**
   - Run multiple application instances behind a load balancer
   - Implement database read replicas for read-heavy workloads
   - Use connection pooling and load balancing

### Resource Sizing

1. **Small Deployment (< 1M traces/day)**
   ```yaml
   resources:
     api:
       cpu: "1"
       memory: "2Gi"
     database:
       cpu: "2"
       memory: "4Gi"
       storage: "100Gi"
   ```

2. **Medium Deployment (1M-10M traces/day)**
   ```yaml
   resources:
     api:
       replicas: 3
       cpu: "2"
       memory: "4Gi"
     database:
       cpu: "4"
       memory: "8Gi"
       storage: "500Gi"
   ```

3. **Large Deployment (> 10M traces/day)**
   ```yaml
   resources:
     api:
       replicas: 5
       cpu: "4"
       memory: "8Gi"
     database:
       cpu: "8"
       memory: "16Gi"
       storage: "1Ti"
   ```

## Best Practices

1. **Always profile before optimizing**
2. **Monitor key metrics continuously**
3. **Use connection pooling for database connections**
4. **Implement proper caching strategies**
5. **Test performance changes in staging environment**
6. **Document performance requirements and SLAs**
7. **Regular performance reviews and optimization cycles**

## Resources

- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [FastAPI Performance Tips](https://fastapi.tiangolo.com/advanced/async-sql-databases/)
- [Locust Documentation](https://docs.locust.io/)
- [Prometheus Monitoring Best Practices](https://prometheus.io/docs/practices/naming/)