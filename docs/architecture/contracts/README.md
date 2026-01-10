# File: docs/architecture/contracts/README.md
# Contract Guidelines

> **DATS Microservices** - How to Define and Use Service Contracts
> 
> This document explains how to create, maintain, and use service contracts
> for both REST APIs (OpenAPI) and event-driven communication (AsyncAPI).

---

## Overview

Contracts are the agreements between services. They define:

- **What** data is exchanged
- **How** services communicate (REST endpoints, events)
- **When** breaking changes occur

### Contract Types

| Type | Format | Purpose |
|------|--------|---------|
| REST API | OpenAPI 3.0+ | Synchronous HTTP endpoints |
| Events | AsyncAPI 2.0+ | Asynchronous message contracts |
| Shared Schemas | JSON Schema | Reusable data models |

---

## Directory Structure

```
docs/architecture/contracts/
├── README.md                    ← You are here
├── schemas/                     # Shared JSON schemas (future)
│   ├── task.json
│   ├── provenance.json
│   └── cloudevents.json

services/{service-name}/
├── contracts/
│   ├── openapi.yaml            # REST API specification
│   └── asyncapi.yaml           # Event specification (if applicable)
```

---

## OpenAPI (REST APIs)

### When to Use

Use OpenAPI for any service that exposes HTTP endpoints:
- Model Gateway (`/generate`, `/models`)
- Orchestration Service (`/tasks`)
- Agent Service (`/analyze`, `/decompose`)

### Template

```yaml
# services/{service}/contracts/openapi.yaml
openapi: 3.0.3
info:
  title: DATS {Service Name} API
  version: 1.0.0
  description: |
    Brief description of what this service does.
    
    ## Authentication
    Currently no authentication required (internal service).
    
    ## Rate Limits
    Default: 100 requests/minute per client.

servers:
  - url: http://{service-name}:8000/api/v1
    description: Internal service endpoint

paths:
  /health:
    get:
      operationId: healthCheck
      summary: Service health check
      tags:
        - Health
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /your-endpoint:
    post:
      operationId: yourOperation
      summary: What this endpoint does
      tags:
        - Main
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/YourRequest'
      responses:
        '200':
          description: Success response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/YourResponse'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

components:
  schemas:
    HealthResponse:
      type: object
      required:
        - status
        - version
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        version:
          type: string
          example: "1.0.0"
        timestamp:
          type: string
          format: date-time

    ErrorResponse:
      type: object
      required:
        - error
      properties:
        error:
          type: object
          required:
            - code
            - message
          properties:
            code:
              type: string
              example: "VALIDATION_ERROR"
            message:
              type: string
              example: "Invalid request body"
            details:
              type: object
              additionalProperties: true

    YourRequest:
      type: object
      required:
        - field1
      properties:
        field1:
          type: string
          description: Description of field1
        field2:
          type: integer
          description: Optional field
          default: 10

    YourResponse:
      type: object
      required:
        - id
        - result
      properties:
        id:
          type: string
          format: uuid
        result:
          type: string
```

### Validation

```bash
# Lint OpenAPI spec
npx @stoplight/spectral-cli lint services/*/contracts/openapi.yaml

# Generate mock server
npx @stoplight/prism-cli mock services/model-gateway/contracts/openapi.yaml
```

---

## AsyncAPI (Events)

### When to Use

Use AsyncAPI for services that publish or subscribe to events:
- Orchestration Service (publishes `task.created`)
- Worker Service (subscribes to `task.ready.{tier}`)
- QA Service (publishes `task.validated`)

### Template

```yaml
# services/{service}/contracts/asyncapi.yaml
asyncapi: 2.6.0
info:
  title: DATS {Service Name} Events
  version: 1.0.0
  description: |
    Event contracts for {Service Name}.
    
    All events use CloudEvents envelope format.

servers:
  rabbitmq:
    url: amqp://192.168.1.49:5672
    protocol: amqp
    description: RabbitMQ message broker

channels:
  task.created:
    description: Published when a new task is submitted
    publish:
      operationId: publishTaskCreated
      message:
        $ref: '#/components/messages/TaskCreated'

  task.ready.{tier}:
    description: Published when task is ready for worker
    parameters:
      tier:
        description: Task tier (tiny, small, large, frontier)
        schema:
          type: string
          enum: [tiny, small, large, frontier]
    subscribe:
      operationId: subscribeTaskReady
      message:
        $ref: '#/components/messages/TaskReady'

components:
  messages:
    TaskCreated:
      name: TaskCreated
      title: Task Created Event
      contentType: application/json
      payload:
        $ref: '#/components/schemas/CloudEvent'
      examples:
        - payload:
            specversion: "1.0"
            type: "dats.task.created"
            source: "/orchestration"
            id: "550e8400-e29b-41d4-a716-446655440000"
            time: "2026-01-09T12:00:00Z"
            datacontenttype: "application/json"
            data:
              task_id: "abc123"
              project_id: "proj-456"
              description: "Create a fibonacci function"

    TaskReady:
      name: TaskReady
      title: Task Ready for Worker
      contentType: application/json
      payload:
        $ref: '#/components/schemas/CloudEvent'

  schemas:
    CloudEvent:
      type: object
      required:
        - specversion
        - type
        - source
        - id
      properties:
        specversion:
          type: string
          const: "1.0"
        type:
          type: string
          pattern: "^dats\\..+"
        source:
          type: string
        id:
          type: string
          format: uuid
        time:
          type: string
          format: date-time
        datacontenttype:
          type: string
          default: "application/json"
        data:
          type: object
          additionalProperties: true
```

### Validation

```bash
# Validate AsyncAPI spec
npx @asyncapi/cli validate services/*/contracts/asyncapi.yaml

# Generate documentation
npx @asyncapi/cli generate fromTemplate services/orchestration/contracts/asyncapi.yaml @asyncapi/html-template -o docs/events/
```

---

## CloudEvents Envelope

All events MUST use CloudEvents format:

```json
{
  "specversion": "1.0",
  "type": "dats.task.created",
  "source": "/orchestration",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "time": "2026-01-09T12:00:00Z",
  "datacontenttype": "application/json",
  "data": {
    "task_id": "abc123",
    "project_id": "proj-456",
    "tier": "small",
    "domain": "code-general"
  }
}
```

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `specversion` | Always "1.0" | `"1.0"` |
| `type` | Event type, prefixed with `dats.` | `"dats.task.created"` |
| `source` | Service that published | `"/orchestration"` |
| `id` | Unique event ID (UUID) | `"550e8400-..."` |

### Optional Fields

| Field | Description |
|-------|-------------|
| `time` | ISO 8601 timestamp |
| `datacontenttype` | Usually `"application/json"` |
| `data` | Event payload |

---

## Versioning

### Semantic Versioning

Contracts follow semver (MAJOR.MINOR.PATCH):

| Change | Version Bump | Example |
|--------|--------------|---------|
| Remove field | MAJOR | 1.0.0 → 2.0.0 |
| Change field type | MAJOR | 1.0.0 → 2.0.0 |
| Add required field | MAJOR | 1.0.0 → 2.0.0 |
| Add optional field | MINOR | 1.0.0 → 1.1.0 |
| Add new endpoint | MINOR | 1.0.0 → 1.1.0 |
| Fix documentation | PATCH | 1.0.0 → 1.0.1 |

### Version in Contract

```yaml
info:
  title: DATS Model Gateway API
  version: 1.2.0  # ← Update this
```

### Breaking Change Process

1. **Announce** - Notify consuming teams
2. **Deprecate** - Mark old fields as deprecated
3. **Dual Support** - Support both old and new for 2 releases
4. **Remove** - Remove old after migration complete

---

## Contract Testing

### Provider Testing

Ensure implementation matches contract:

```python
# tests/contract/test_openapi_compliance.py
import schemathesis

schema = schemathesis.from_path("contracts/openapi.yaml")

@schema.parametrize()
def test_api_compliance(case):
    """Test that API matches OpenAPI spec."""
    response = case.call_asgi(app)
    case.validate_response(response)
```

### Consumer Testing

Ensure consumers can handle provider responses:

```python
# Using Pact for consumer-driven contracts
from pact import Consumer, Provider

pact = Consumer('worker-service').has_pact_with(Provider('model-gateway'))

@pact.given('model gemma3:4b is available')
@pact.upon_receiving('a generate request')
def test_generate_request():
    # Define expected interaction
    pass
```

---

## Tools

### Recommended Tooling

| Tool | Purpose | Install |
|------|---------|---------|
| Spectral | OpenAPI linting | `npm i -g @stoplight/spectral-cli` |
| Prism | OpenAPI mocking | `npm i -g @stoplight/prism-cli` |
| AsyncAPI CLI | AsyncAPI validation | `npm i -g @asyncapi/cli` |
| Schemathesis | Contract testing | `pip install schemathesis` |

### Makefile Targets

Each service should have:

```makefile
# Makefile
.PHONY: contract-lint contract-test contract-mock

contract-lint:
	npx @stoplight/spectral-cli lint contracts/openapi.yaml
	npx @asyncapi/cli validate contracts/asyncapi.yaml

contract-test:
	pytest tests/contract/

contract-mock:
	npx @stoplight/prism-cli mock contracts/openapi.yaml -p 4010
```

---

## Related Documents

- [ADR-005: Contract Strategy](decisions/005-contract-strategy.md) - Why we chose this approach
- [SERVICE_DONE_DEFINITION.md](SERVICE_DONE_DEFINITION.md) - Contract requirements for PRs
- [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md) - Overall architecture

---

*Contracts are the source of truth for service interfaces. Implementation must match contracts.*