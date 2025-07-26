# GitHub Actions CI Workflow

Since I cannot create GitHub Actions workflows directly due to permission restrictions, please manually create the following CI workflow file.

## Create `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: llm_metrics_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true
    
    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --no-interaction --no-root
    
    - name: Install project
      run: poetry install --no-interaction
    
    - name: Run linting
      run: |
        poetry run black --check .
        poetry run isort --check-only .
        poetry run flake8 .
    
    - name: Run type checking
      run: poetry run mypy src/
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/llm_metrics_test
      run: poetry run pytest --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t llm-cost-tracker:test .
    
    - name: Test Docker image
      run: |
        docker run --rm -d --name test-container -p 8000:8000 llm-cost-tracker:test
        sleep 10
        curl -f http://localhost:8000/health || exit 1
        docker stop test-container
```

## Instructions

1. Create the directory: `.github/workflows/`
2. Save the above content as `.github/workflows/ci.yml`
3. Commit the workflow file to enable automated testing

This workflow will:
- Run tests on Python 3.11 and 3.12
- Test against PostgreSQL database
- Run linting, type checking, and tests
- Build and test the Docker image
- Upload coverage reports to Codecov