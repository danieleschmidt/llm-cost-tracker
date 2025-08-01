# Testing Guide for LLM Cost Tracker

This guide covers the comprehensive testing strategy and practices for the LLM Cost Tracker project.

## Testing Architecture

Our testing strategy follows a pyramid approach with four main layers:

1. **Unit Tests** - Fast, isolated tests for individual components
2. **Integration Tests** - Tests for component interactions
3. **End-to-End Tests** - Full workflow testing
4. **Performance Tests** - Load and benchmark testing

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data and mock objects
│   ├── __init__.py
│   ├── sample_data.py       # Sample data for testing
│   └── mocks.py            # Mock objects and utilities
├── unit/                   # Unit tests
│   └── __init__.py
├── integration/            # Integration tests
│   └── __init__.py
├── e2e/                   # End-to-end tests
│   ├── __init__.py
│   └── test_full_workflow.py
├── performance/           # Performance tests
│   ├── __init__.py
│   └── test_benchmarks.py
└── load/                  # Load testing
    └── test_api_performance.py
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m e2e               # End-to-end tests only
pytest -m performance       # Performance tests only

# Run tests with coverage
make test-coverage
```

### Advanced Test Options

```bash
# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_database.py

# Run tests matching pattern
pytest -k "test_cost_calculation"

# Run tests with specific markers
pytest -m "unit and not slow"
```

## Test Categories and Markers

We use pytest markers to categorize tests:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.security` - Security-related tests
- `@pytest.mark.external` - Tests requiring external services

## Writing Tests

### Unit Test Example

```python
import pytest
from llm_cost_tracker.cost_calculator import CostCalculator

@pytest.mark.unit
def test_cost_calculation_openai():
    calculator = CostCalculator()
    
    cost = calculator.calculate_cost(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50
    )
    
    expected_cost = (100 * 0.03 / 1000) + (50 * 0.06 / 1000)
    assert cost == expected_cost
```

### Integration Test Example

```python
import pytest
from tests.fixtures.mocks import MockDatabase

@pytest.mark.integration
async def test_cost_tracking_workflow(test_client, mock_database):
    # Test the full cost tracking workflow
    response = await test_client.post("/api/v1/track", json={
        "model": "gpt-4",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "user_id": "test_user"
    })
    
    assert response.status_code == 200
    assert len(mock_database.records) == 1
    assert mock_database.records[0]["model"] == "gpt-4"
```

### Performance Test Example

```python
import pytest
import asyncio
from tests.fixtures.sample_data import generate_performance_data

@pytest.mark.performance
@pytest.mark.benchmark
def test_bulk_insert_performance(benchmark, test_db_session):
    data = generate_performance_data(1000)
    
    async def bulk_insert():
        for record in data:
            await test_db_session.add(record)
        await test_db_session.commit()
    
    # Benchmark should complete in under 5 seconds for 1000 records
    result = benchmark(asyncio.run, bulk_insert())
    assert result is not None
```

## Test Fixtures

### Database Fixtures

```python
@pytest.fixture
async def test_db_session():
    """Provides a clean database session for each test."""
    # Creates isolated database session
    pass

@pytest.fixture
def sample_cost_data():
    """Provides sample cost tracking data."""
    # Returns standardized test data
    pass
```

### Mock Fixtures

```python
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    # Returns configured mock client
    pass

@pytest.fixture
def mock_slack_webhook():
    """Mock Slack webhook for notification testing."""
    # Returns mock webhook handler
    pass
```

## Mocking Strategy

### External Services

We mock all external services to ensure:
- Tests run without network dependencies
- Consistent test results
- Fast test execution
- Cost control (no real API calls)

```python
# Mock LLM providers
@pytest.fixture
def mock_llm_providers(monkeypatch):
    monkeypatch.setattr("openai.ChatCompletion.create", mock_openai_response)
    monkeypatch.setattr("anthropic.Client.messages.create", mock_anthropic_response)
```

### Database Mocking

For unit tests, we use in-memory SQLite:

```python
@pytest.fixture
def test_settings():
    return Settings(database_url="sqlite+aiosqlite:///:memory:")
```

## Test Data Management

### Sample Data

Use the fixtures in `tests/fixtures/sample_data.py`:

```python
from tests.fixtures.sample_data import SAMPLE_COST_RECORDS, MODEL_PRICING

def test_with_sample_data():
    records = SAMPLE_COST_RECORDS
    pricing = MODEL_PRICING["gpt-4"]
    # Test logic here
```

### Dynamic Data Generation

For performance tests, generate data dynamically:

```python
from tests.fixtures.sample_data import generate_performance_data

def test_performance():
    data = generate_performance_data(count=10000)
    # Performance test logic
```

## Performance Testing

### Benchmarking

We use `pytest-benchmark` for performance testing:

```python
@pytest.mark.performance
def test_cost_calculation_performance(benchmark):
    def calculate_many_costs():
        calculator = CostCalculator()
        for _ in range(1000):
            calculator.calculate_cost("gpt-4", 100, 50)
    
    benchmark(calculate_many_costs)
```

### Load Testing

For API load testing, we use Locust:

```python
# tests/load/locustfile.py
from locust import HttpUser, task

class CostTrackerUser(HttpUser):
    @task
    def track_cost(self):
        self.client.post("/api/v1/track", json={
            "model": "gpt-4",
            "prompt_tokens": 100,
            "completion_tokens": 50
        })
```

## Security Testing

### Input Validation Tests

```python
@pytest.mark.security
def test_sql_injection_protection(test_client):
    malicious_input = "'; DROP TABLE cost_records; --"
    
    response = test_client.post("/api/v1/track", json={
        "user_id": malicious_input
    })
    
    # Should reject malicious input
    assert response.status_code == 400
```

### Authentication Tests

```python
@pytest.mark.security
def test_unauthorized_access(test_client):
    response = test_client.get("/api/v1/admin/metrics")
    assert response.status_code == 401
```

## Continuous Integration

### Test Pipeline

Our CI pipeline runs tests in this order:

1. **Lint and Format Checks**
   ```bash
   make lint
   make typecheck
   ```

2. **Security Scanning**
   ```bash
   make security
   ```

3. **Unit Tests**
   ```bash
   pytest -m unit --cov=src/llm_cost_tracker
   ```

4. **Integration Tests**
   ```bash
   pytest -m integration
   ```

5. **Performance Regression Tests**
   ```bash
   pytest -m performance --benchmark-only
   ```

### Coverage Requirements

- **Unit Tests**: >90% coverage
- **Integration Tests**: >80% coverage
- **Overall Project**: >85% coverage

## Test Environment Setup

### Local Development

```bash
# Setup test environment
make setup-dev

# Install test dependencies
poetry install --with dev

# Setup pre-commit hooks (includes test running)
pre-commit install
```

### Docker Testing

```bash
# Run tests in Docker environment
docker-compose -f docker-compose.test.yml up --build

# Run specific test suite
docker-compose -f docker-compose.test.yml run tests pytest -m unit
```

## Best Practices

### Test Writing Guidelines

1. **Test Naming**: Use descriptive names that explain what is being tested
   ```python
   def test_cost_calculation_with_gpt4_returns_correct_amount():
   ```

2. **Arrange-Act-Assert**: Structure tests clearly
   ```python
   def test_example():
       # Arrange
       calculator = CostCalculator()
       
       # Act
       result = calculator.calculate(model="gpt-4", tokens=100)
       
       # Assert
       assert result == expected_value
   ```

3. **Single Responsibility**: Each test should verify one specific behavior

4. **Isolation**: Tests should not depend on each other

5. **Fast Feedback**: Unit tests should complete in <100ms

### Mock Guidelines

1. **Mock at Boundaries**: Mock external systems, not internal logic
2. **Realistic Mocks**: Mock responses should match real API responses
3. **State Verification**: Verify both return values and side effects
4. **Reset Mocks**: Clean up mock state between tests

### Performance Test Guidelines

1. **Baseline Metrics**: Establish performance baselines
2. **Regression Detection**: Fail tests if performance degrades >20%
3. **Resource Monitoring**: Monitor memory and CPU usage
4. **Realistic Load**: Use realistic data volumes and patterns

## Troubleshooting

### Common Issues

1. **Async Test Issues**
   ```python
   # Use pytest-asyncio
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

2. **Database Connection Issues**
   ```python
   # Ensure proper cleanup
   @pytest.fixture
   async def db_session():
       session = create_session()
       try:
           yield session
       finally:
           await session.close()
   ```

3. **Mock Not Working**
   ```python
   # Use monkeypatch for reliable mocking
   def test_with_mock(monkeypatch):
       monkeypatch.setattr("module.function", mock_function)
   ```

### Debugging Tests

```bash
# Run with debugger
pytest --pdb

# Capture output
pytest -s

# Show local variables on failure
pytest --tb=long
```

## Test Metrics and Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src/llm_cost_tracker --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Performance Reports

```bash
# Generate performance benchmark report
pytest --benchmark-only --benchmark-json=benchmark.json
```

### Test Result Analysis

We track:
- Test execution time trends
- Coverage percentage over time  
- Performance regression metrics
- Flaky test identification
- Security test results

## Integration with Development Workflow

### Pre-commit Hooks

Tests are automatically run on commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest -m unit
        language: system
        pass_filenames: false
```

### IDE Integration

VS Code configuration for testing:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

This comprehensive testing approach ensures high code quality, reliable functionality, and maintainable test suites.