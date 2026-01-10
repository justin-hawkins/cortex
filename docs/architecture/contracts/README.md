# File: docs/architecture/contracts/README.md
# Contract Guidelines

> **DATS Microservices** - Service Contract Specifications

---

## Overview

Contracts define **what** data is exchanged and **how** services communicate.

| Type | Format | Purpose |
|------|--------|---------|
| REST API | OpenAPI 3.0+ | Synchronous HTTP endpoints |
| Events | AsyncAPI 2.0+ | Asynchronous message contracts |
| Shared Schemas | JSON Schema | Reusable data models |

---

## Directory Structure

```
services/{service-name}/contracts/
├── openapi.yaml    # REST API specification
└── asyncapi.yaml   # Event specification (if applicable)
```

---

## OpenAPI Quick Reference

```yaml
openapi: 3.0.3
info:
  title: DATS {Service Name} API
  version: 1.0.0
servers:
  - url: http://{service-name}:8000/api/v1
paths:
  /health:
    get:
      operationId: healthCheck
      responses:
        '200':
          description: Service is healthy
```

---

## AsyncAPI Quick Reference

```yaml
asyncapi: 2.6.0
info:
  title: DATS {Service Name} Events
  version: 1.0.0
channels:
  task.created:
    publish:
      message:
        $ref: '#/components/messages/TaskCreated'
```

---

## CloudEvents Format

All messages use CloudEvents envelope:

```json
{
  "specversion": "1.0",
  "type": "dats.task.created",
  "source": "/orchestration",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "time": "2026-01-09T12:00:00Z",
  "data": {"task_id": "abc123"}
}
```

---

## Versioning Rules

| Change | Version Bump |
|--------|--------------|
| Remove/change field type | MAJOR |
| Add required field | MAJOR |
| Add optional field | MINOR |
| Add new endpoint | MINOR |
| Fix documentation | PATCH |

---

## Validation Commands

```bash
# Lint OpenAPI
npx @stoplight/spectral-cli lint contracts/openapi.yaml

# Validate AsyncAPI
npx @asyncapi/cli validate contracts/asyncapi.yaml

# Contract tests
pytest tests/contract/
```

---

## Related Documents

- [ADR-005: Contract Strategy](decisions/005-contract-strategy.md)
- [SERVICE_DONE_DEFINITION.md](SERVICE_DONE_DEFINITION.md)
