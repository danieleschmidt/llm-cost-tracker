# 🔌 Quantum Task Planner API Reference

## Base URL
```
https://api.quantum-planner.your-domain.com
```

## Authentication
Currently using API key authentication. Include in headers:
```http
Authorization: Bearer YOUR_API_KEY
X-API-Key: YOUR_API_KEY
```

## Content Type
All requests should use JSON:
```http
Content-Type: application/json
```

## 📋 Core Task Management

### Create Task
Create a new quantum task.

**Endpoint:** `POST /api/v1/quantum/tasks`

**Request Body:**
```json
{
  "id": "unique_task_id",
  "name": "Task Name",
  "description": "Task description",
  "priority": 8.5,
  "estimated_duration_minutes": 30,
  "required_resources": {
    "cpu_cores": 2.0,
    "memory_gb": 4.0,
    "storage_gb": 10.0
  },
  "dependencies": ["prerequisite_task_id"]
}
```

**Response:**
```json
{
  "id": "unique_task_id",
  "name": "Task Name",
  "description": "Task description",
  "state": "superposition",
  "priority": 8.5,
  "execution_probability": 0.95,
  "created_at": "2024-01-15T10:30:00Z",
  "dependencies": ["prerequisite_task_id"],
  "entangled_tasks": [],
  "error_message": null
}
```

### List Tasks
Retrieve all quantum tasks.

**Endpoint:** `GET /api/v1/quantum/tasks`

**Query Parameters:**
- `state` (optional): Filter by task state
- `priority_min` (optional): Minimum priority level
- `priority_max` (optional): Maximum priority level
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
[
  {
    "id": "task_1",
    "name": "Task 1",
    "state": "completed",
    "priority": 9.0,
    "execution_probability": 1.0,
    "created_at": "2024-01-15T10:30:00Z",
    "completed_at": "2024-01-15T10:35:00Z"
  }
]
```

### Get Task Details
Retrieve detailed information about a specific task.

**Endpoint:** `GET /api/v1/quantum/tasks/{task_id}`

**Response:**
```json
{
  "id": "task_id",
  "name": "Task Name", 
  "description": "Detailed task description",
  "state": "executing",
  "priority": 7.5,
  "execution_probability": 0.87,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:35:00Z",
  "completed_at": null,
  "dependencies": ["dep1", "dep2"],
  "entangled_tasks": ["entangled_task_1"],
  "required_resources": {
    "cpu_cores": 1.5,
    "memory_gb": 2.0
  },
  "interference_pattern": {
    "nearby_task": 0.05,
    "conflicting_task": -0.1
  },
  "error_message": null
}
```

### Update Task
Update an existing task's properties.

**Endpoint:** `PUT /api/v1/quantum/tasks/{task_id}`

**Request Body:**
```json
{
  "name": "Updated Task Name",
  "priority": 9.0,
  "required_resources": {
    "cpu_cores": 3.0,
    "memory_gb": 6.0
  }
}
```

### Delete Task
Remove a task from the system.

**Endpoint:** `DELETE /api/v1/quantum/tasks/{task_id}`

**Response:**
```json
{
  "message": "Task task_id deleted successfully"
}
```

## 📅 Scheduling & Execution

### Generate Optimal Schedule
Generate an optimized task execution schedule using quantum annealing.

**Endpoint:** `GET /api/v1/quantum/schedule`

**Query Parameters:**
- `max_iterations` (optional): Annealing iterations (default: 1000)
- `include_metrics` (optional): Include scheduling metrics (default: false)

**Response:**
```json
{
  "schedule": ["task1", "task3", "task2", "task4"],
  "estimated_total_duration_minutes": 125.5,
  "resource_conflicts": 0,
  "dependency_violations": 0,
  "optimization_metrics": {
    "initial_cost": 450.7,
    "final_cost": 123.2,
    "improvement_ratio": 0.726,
    "iterations_used": 847
  }
}
```

### Execute Tasks
Execute tasks according to optimal or custom schedule.

**Endpoint:** `POST /api/v1/quantum/execute`

**Request Body:**
```json
{
  "schedule": ["task1", "task2", "task3"],
  "execution_mode": "parallel",
  "max_concurrent": 5
}
```

**Response:**
```json
{
  "total_tasks": 3,
  "successful_tasks": 3,
  "failed_tasks": 0,
  "success_rate": 1.0,
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T10:45:00Z",
  "parallel_batches": 2,
  "task_results": {
    "task1": true,
    "task2": true,
    "task3": true
  }
}
```

## 📊 System Monitoring

### System State
Get current quantum planning system state.

**Endpoint:** `GET /api/v1/quantum/system/state`

**Response:**
```json
{
  "total_tasks": 15,
  "resource_utilization": {
    "cpu": 0.65,
    "memory": 0.45,
    "storage": 0.25,
    "bandwidth": 0.12
  },
  "task_states": {
    "task1": {
      "state": "completed",
      "probability": 1.0,
      "entangled_with": [],
      "dependencies": []
    }
  },
  "execution_history_size": 142
}
```

### Health Check
Basic system health endpoint.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "llm-cost-tracker",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Detailed Health Check
Comprehensive health status with all subsystems.

**Endpoint:** `GET /health/detailed`

**Response:**
```json
{
  "status": "healthy",
  "service": "quantum-task-planner",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15.2
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.98,
      "size": 1024
    },
    "scheduler": {
      "status": "healthy",
      "active_tasks": 5
    }
  }
}
```

### Resource Status
Get current resource pool state.

**Endpoint:** `GET /api/v1/quantum/system/resources`

**Response:**
```json
{
  "total_resources": {
    "cpu_cores": 8.0,
    "memory_gb": 16.0,
    "storage_gb": 500.0,
    "network_bandwidth": 1000.0
  },
  "allocated_resources": {
    "cpu_cores": 3.5,
    "memory_gb": 8.2,
    "storage_gb": 125.0,
    "network_bandwidth": 250.0
  },
  "available_resources": {
    "cpu_cores": 4.5,
    "memory_gb": 7.8,
    "storage_gb": 375.0,
    "network_bandwidth": 750.0
  },
  "utilization_percentage": {
    "cpu": 43.75,
    "memory": 51.25,
    "storage": 25.0,
    "bandwidth": 25.0
  }
}
```

## 🎛️ Performance Metrics

### Cache Metrics
Get cache performance statistics.

**Endpoint:** `GET /metrics/cache`

**Response:**
```json
{
  "cache_stats": {
    "size": 856,
    "max_size": 1000,
    "total_size_bytes": 2048576,
    "hits": 9456,
    "misses": 234,
    "evictions": 12,
    "hit_rate": 0.976,
    "running": true
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Concurrency Metrics
Get concurrency and task queue metrics.

**Endpoint:** `GET /metrics/concurrency`

**Response:**
```json
{
  "task_queue_stats": {
    "running": true,
    "worker_count": 8,
    "completed_tasks": 1234,
    "failed_tasks": 15,
    "retried_tasks": 45,
    "primary_queue_size": 5,
    "overflow_queue_size": 0,
    "circuit_breakers": {
      "task_execution": false,
      "database": false
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Auto-Scaling Metrics
Get auto-scaling configuration and history.

**Endpoint:** `GET /metrics/auto-scaling`

**Response:**
```json
{
  "configuration": {
    "min_workers": 2,
    "max_workers": 20,
    "current_workers": 8,
    "target_queue_size": 5
  },
  "scaling_history": [
    {
      "timestamp": "2024-01-15T10:25:00Z",
      "action": "scale_up",
      "from_workers": 6,
      "to_workers": 8,
      "reason": "high_queue_utilization"
    }
  ],
  "recent_metrics_count": 120,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🌐 Internationalization

### Set Language
Change the system language for responses.

**Endpoint:** `POST /api/v1/i18n/language`

**Request Body:**
```json
{
  "language": "es"
}
```

**Supported Languages:**
- `en` - English
- `es` - Spanish  
- `fr` - French
- `de` - German
- `ja` - Japanese
- `zh` - Chinese (Simplified)

### Get Available Languages
List all supported languages.

**Endpoint:** `GET /api/v1/i18n/languages`

**Response:**
```json
{
  "current_language": "en",
  "available_languages": [
    {"code": "en", "name": "English"},
    {"code": "es", "name": "Español"},
    {"code": "fr", "name": "Français"},
    {"code": "de", "name": "Deutsch"},
    {"code": "ja", "name": "日本語"},
    {"code": "zh", "name": "中文"}
  ]
}
```

## 🛡️ Compliance & Privacy

### Data Subject Rights (GDPR Article 15)
Get all data for a specific data subject.

**Endpoint:** `GET /api/v1/compliance/data-subject/{subject_id}`

**Response:**
```json
{
  "data_subject_id": "user123",
  "processing_records": [
    {
      "record_id": "rec_001",
      "timestamp": "2024-01-15T10:30:00Z",
      "processing_purpose": "Task optimization",
      "data_categories": ["pii", "internal"],
      "legal_basis": "legitimate_interest"
    }
  ],
  "consent_records": [
    {
      "consent_id": "consent_001",
      "timestamp": "2024-01-15T10:00:00Z",
      "consent_given": true,
      "purposes": ["analytics", "optimization"]
    }
  ]
}
```

### Data Deletion (GDPR Article 17)
Delete all data for a specific data subject.

**Endpoint:** `DELETE /api/v1/compliance/data-subject/{subject_id}`

**Response:**
```json
{
  "data_subject_id": "user123",
  "deleted_processing_records": 5,
  "deleted_consent_records": 1,
  "deletion_timestamp": "2024-01-15T10:30:00Z"
}
```

### Privacy Impact Assessment
Generate privacy impact assessment report.

**Endpoint:** `GET /api/v1/compliance/privacy-impact-assessment`

**Response:**
```json
{
  "assessment_date": "2024-01-15T10:30:00Z",
  "data_categories_processed": ["pii", "internal", "sensitive"],
  "risk_level": "medium",
  "risks_identified": [
    "Processing of personally identifiable information",
    "Automated decision making processes"
  ],
  "mitigation_measures": [
    "Data minimization principles applied",
    "Anonymization and pseudonymization where possible"
  ],
  "compliance_status": {
    "eu_gdpr": {
      "compliant": true,
      "notes": "Basic compliance measures implemented"
    }
  }
}
```

## 🔧 Administrative

### System Reset
Reset the quantum planning system (admin only).

**Endpoint:** `POST /api/v1/quantum/system/reset`

**Response:**
```json
{
  "message": "System reset successfully"
}
```

### Demo Mode
Run quantum planning demonstration.

**Endpoint:** `GET /api/v1/quantum/demo`

**Response:**
```json
{
  "message": "Demo completed successfully",
  "execution_results": {
    "total_tasks": 4,
    "successful_tasks": 4,
    "success_rate": 1.0
  },
  "system_state": {
    "total_tasks": 4,
    "resource_utilization": {
      "cpu": 0.15,
      "memory": 0.22
    }
  }
}
```

## 📈 Prometheus Metrics

Standard Prometheus metrics available at `/metrics`:

### Core Metrics
- `quantum_tasks_total{state="completed|failed|executing"}` - Total tasks by state
- `quantum_execution_duration_seconds` - Task execution time histogram
- `quantum_schedule_optimization_duration_seconds` - Schedule generation time
- `quantum_cache_hits_total` - Cache hit counter
- `quantum_cache_misses_total` - Cache miss counter
- `quantum_resource_utilization_ratio{resource="cpu|memory|storage"}` - Resource usage

### Business Metrics
- `quantum_success_rate` - Overall task success rate
- `quantum_schedule_quality_score` - Quality of generated schedules
- `quantum_entanglement_effects_total` - Quantum entanglement applications
- `quantum_interference_patterns_total` - Interference pattern applications

## ❌ Error Handling

### Standard Error Response
```json
{
  "error": "Validation Error",
  "message": "Task priority must be between 1 and 10",
  "request_id": "req_12345",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "field": "priority",
    "value": 15,
    "constraint": "range(1, 10)"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request (validation error)
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict (duplicate resource)
- `422` - Unprocessable Entity
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Types
- `ValidationError` - Input validation failed
- `SecurityValidationError` - Security check failed
- `DatabaseConnectionError` - Database connectivity issue
- `DatabaseOperationError` - Database operation failed
- `CircuitBreakerOpenError` - Circuit breaker preventing operation
- `ResourceExhaustionError` - Insufficient resources
- `DependencyError` - Task dependency issue

## 🚀 Rate Limiting

API endpoints are rate limited:
- **Default**: 100 requests/minute per API key
- **Burst**: Up to 10 requests/second
- **Headers**: 
  - `X-RateLimit-Limit` - Request limit
  - `X-RateLimit-Remaining` - Remaining requests
  - `X-RateLimit-Reset` - Reset timestamp

## 📝 Changelog

### v0.1.0
- Initial quantum task planner API
- Core task management endpoints
- Quantum annealing scheduler
- Real-time monitoring
- I18n support
- GDPR compliance features

---

*For additional examples and advanced usage patterns, see the `/examples` directory in the repository.*