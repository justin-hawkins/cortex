# File: docs/architecture/decisions/005-contract-strategy.md
# ADR 005: Service Contract Strategy

**Status**: Accepted  
**Date**: January 2026  
**Deciders**: Architecture Team

---

## Context

As we build out the microservices architecture, different teams will own different services. We need a contract mechanism that:

1. Defines clear interfaces between services
2. Enables independent development and deployment
3. Catches breaking changes before production
4. Supports both synchronous (REST) and asynchronous (events) communication

### Options Considered

1. **OpenAPI only** - Just document REST APIs
2. **Code-based contracts** - Shared Pydantic models in dats-common
3. **OpenAPI + AsyncAPI** - Full specification coverage
4. **OpenAPI + AsyncAPI + Contract Testing** - Specs plus automated validation

## Decision

We will use **Option 4: OpenAPI + AsyncAPI + Contract Testing**.

### Contract Types

| Communication Pattern | Contract Format | Tooling |
|-----------------------|-----------------|---------|
| REST APIs | OpenAPI 3.0+ | Spectral (lint), Prism (mock) |
| Event Messages | AsyncAPI 2.0+ | AsyncAPI CLI (validate) |
| Shared Data Models | JSON Schema | Referenced by OpenAPI/AsyncAPI |

## Rationale

### Why OpenAPI for REST

- Industry standard, excellent tooling
- Auto-generate client SDKs
- Built-in validation and mocking
- Swagger UI for documentation

### Why AsyncAPI for Events

- OpenAPI equivalent for async APIs
- Documents exchange topology, routing keys
- CloudEvents envelope structure
- Publisher/subscriber relationships clear

### Why Contract Testing

- Catches breaking changes in CI before deployment
- Consumer-driven: consuming teams define expectations
- Provider validation: implementation matches spec
- Enables confident independent deployment

## Contract-First Development Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTRACT-FIRST WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DESIGN                                                       │
│     └── Write OpenAPI/AsyncAPI spec                             │
│         └── Team reviews contract                                │
│                                                                  │
│  2. MOCK                                                         │
│     └── Generate mock server from spec                          │
│         └── Consumers develop against mock                       │
│                                                                  │
│  3. IMPLEMENT                                                    │
│     └── Provider implements to match contract                   │
│         └── Contract tests validate implementation               │
│                                                                  │
│  4. DEPLOY                                                       │
│     └── CI validates no breaking changes                        │
│         └── Deploy with confidence                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Contract File Locations

### Per-Service Contracts

```
services/{service-name}/
├── contracts/
│   ├── openapi.yaml      # REST API specification
│   └── asyncapi.yaml     # Event specification (if applicable)
```

### Shared Schemas

```
docs/architecture/contracts/
├── README.md              # Contract guidelines
├── schemas/               # Shared JSON schemas
│   ├── task.json
│   ├── provenance.json
│   └── cloudevents.json
```

## Semantic Versioning for Contracts

Contracts follow semantic versioning:

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Breaking (removed field, type change) | MAJOR | 1.0.0 → 2.0.0 |
| Additive (new optional field/endpoint) | MINOR | 1.0.0 → 1.1.0 |
| Fix (docs, examples only) | PATCH | 1.0.0 → 1.0.1 |

### Breaking Change Policy

- Contracts must be backward compatible for **2 release cycles**
- Breaking changes require:
  - ADR documenting the change
  - Consumer migration plan
  - Deprecation warnings in prior release
  - Coordinated rollout

## CI/CD Integration

### Contract Validation Pipeline

```yaml
# .github/workflows/contract-validation.yml
on:
  pull_request:
    paths:
      - 'services/*/contracts/**'
      - 'docs/architecture/contracts/**'

jobs:
  validate-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Validate OpenAPI specs
        run: |
          npx @stoplight/spectral-cli lint 'services/*/contracts/openapi.yaml'
      
      - name: Validate AsyncAPI specs
        run: |
          npx @asyncapi/cli validate 'services/*/contracts/asyncapi.yaml'
      
      - name: Check breaking changes
        run: |
          # Compare against main branch
          git fetch origin main
          ./scripts/contract-diff.sh origin/main HEAD
```

### Contract Tests in Service CI

```yaml
# Per-service pipeline
- name: Contract Tests
  run: |
    # Validate implementation matches contract
    make contract-test
```

## Consequences

### Positive

- Clear contracts between teams
- Independent development and deployment possible
- Breaking changes caught before production
- Self-documenting APIs
- Consumers can develop against mocks

### Negative

- Additional files to maintain
- Learning curve for AsyncAPI
- CI pipeline complexity
- Must keep contracts in sync with implementation

## Implementation Requirements

### For Each Service

1. Create `contracts/openapi.yaml` for REST endpoints
2. Create `contracts/asyncapi.yaml` if service publishes/subscribes events
3. Add contract validation to CI pipeline
4. Reference shared schemas where applicable

### Definition of Done

All service changes must comply with [SERVICE_DONE_DEFINITION.md](../SERVICE_DONE_DEFINITION.md), which includes contract requirements.

---

## Related Decisions

- [ADR-001: Repository Strategy](001-repo-strategy.md) - Monorepo structure
- [ADR-003: API Format Selection](003-api-format.md) - REST first, gRPC later
- [MICROSERVICES_DESIGN.md](../MICROSERVICES_DESIGN.md) - Overall architecture