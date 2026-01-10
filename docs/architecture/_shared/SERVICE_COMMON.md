# File: docs/architecture/_shared/SERVICE_COMMON.md
# Service Common Patterns

> **DATS Microservices** - Shared patterns and templates for all services.
> 
> This document contains boilerplate that applies to ALL services. Individual 
> service docs reference this to avoid duplication.

---

## Standard Folder Structure

Every service follows this structure (per [ADR-001](../decisions/001-repo-strategy.md)):

```
services/{service-name}/
├── Dockerfile                 # Self-contained build
├── docker-compose.yml         # Local dev with dependencies
├── pyproject.toml             # Dependencies (references dats-common)
├── requirements.txt           # Locked deps for reproducible builds
├── Makefile                   # build, test, lint, contract-test
├── README.md                  # Setup instructions
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Service settings
│   ├── routers/               # API route handlers
│   └── services/              # Business logic
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Tests with real DB, mocked externals
│   └── contract/              # OpenAPI/AsyncAPI compliance tests
├── config/                    # Service-specific configuration
└── contracts/
    ├── openapi.yaml           # REST API spec (if applicable)
    └── asyncapi.yaml          # Event spec (if applicable)
```

---

## Infrastructure References

All services use centralized infrastructure defined in [`servers.yaml`](../servers.yaml):

| Resource | Endpoint | Purpose |
|----------|----------|---------|
| Ollama (CPU) | `192.168.1.11:11434` | Large coding models (qwen3-coder) |
| Ollama (GPU) | `192.168.1.12:11434` | General inference, embeddings |
| vLLM | `192.168.1.11:8000/v1` | High-throughput inference |
| RabbitMQ | `192.168.1.49:5672` | Message bus |
| Redis | `192.168.1.44:6379` | Caching, rate limiting |

---

## Standard Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dats-common
COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Common Dependencies

All services include these base dependencies:

```
# requirements.txt (base)
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Observability
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
opentelemetry-exporter-otlp>=1.21.0
prometheus-client>=0.19.0

# Messaging (if using events)
pika>=1.3.0
aio-pika>=9.3.0
```

---

## Observability Patterns

### Traces (OpenTelemetry)

```python
from opentelemetry import trace
tracer = trace.get_tracer("service-name")

@tracer.start_as_current_span("operation_name")
async def my_operation():
    span = trace.get_current_span()
    span.set_attribute("custom.attribute", "value")
```

### Metrics (Prometheus)

Standard metrics all services should expose:
```
# Request metrics
{service}_requests_total{endpoint, status}
{service}_request_latency_seconds{endpoint, quantile}

# Dependency health
{service}_dependency_status{dependency}
```

### Logs (Structured JSON)

```json
{
  "timestamp": "2026-01-09T18:00:00Z",
  "level": "INFO",
  "service": "service-name",
  "trace_id": "...",
  "span_id": "...",
  "event": "operation_complete",
  "custom_field": "value"
}
```

---

## Health Check Endpoint

All services must implement `GET /api/v1/health`:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-09T18:00:00Z"
}
```

Optional detailed health at `GET /api/v1/health/detailed`:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "model_gateway": "connected",
    "rabbitmq": "connected",
    "database": "connected"
  }
}
```

---

## Common Environment Variables

```bash
# Telemetry (all services)
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_SERVICE_NAME={service-name}
LOG_LEVEL=INFO

# Infrastructure (from servers.yaml)
RABBITMQ_HOST=192.168.1.49
RABBITMQ_PORT=5672
REDIS_URL=redis://192.168.1.44:6379/0

# Service port
SERVICE_PORT=8000
```

---

## Testing Patterns

### Unit Tests
```python
# tests/unit/test_*.py
import pytest

def test_business_logic():
    """Fast, isolated, mock all dependencies."""
    pass
```

### Integration Tests
```python
# tests/integration/test_*.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_endpoint():
    """Test with real DB, mocked external services."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
```

### Contract Tests
```python
# tests/contract/test_openapi.py
import schemathesis
schema = schemathesis.from_path("contracts/openapi.yaml")

@schema.parametrize()
def test_api_compliance(case):
    """Validate implementation matches OpenAPI spec."""
    response = case.call_asgi(app)
    case.validate_response(response)
```

---

## Makefile Targets

All services should support:

```makefile
.PHONY: build test lint run contract-test

build:
	docker build -t dats-{service-name}:latest .

test:
	pytest tests/unit tests/integration -v

lint:
	ruff check src/ tests/
	mypy src/

run:
	uvicorn src.main:app --reload --port 8000

contract-test:
	pytest tests/contract/
	npx @stoplight/spectral-cli lint contracts/openapi.yaml
```

---

## Contract Specifications

See [Contract Guidelines](../contracts/README.md) for:
- OpenAPI template and requirements
- AsyncAPI template for event-driven services
- CloudEvents envelope format
- Versioning rules

---

## Definition of Done

All service changes must comply with [SERVICE_DONE_DEFINITION.md](../SERVICE_DONE_DEFINITION.md):
- Contract updated (if API changed)
- Tests pass (unit + integration + contract)
- Documentation current
- Health check works
- No cross-service imports

---

*This shared template reduces duplication across service docs. Each service doc contains only unique content.*