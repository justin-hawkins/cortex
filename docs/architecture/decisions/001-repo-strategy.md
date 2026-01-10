# ADR 001: Repository Strategy

**Status**: Accepted  
**Date**: January 2026  
**Deciders**: Architecture Team

---

## Context

As we move from monolith to microservices, we need to decide how to organize our code repositories.

### Options Considered

1. **Monorepo**: All services in one repository
2. **Multi-repo**: Each service in its own repository
3. **Hybrid**: Monorepo initially, split later

## Decision

We will use **Option 3: Monorepo initially, split later**.

### Structure

```
cortex/
├── dats/                     # Current monolith (deprecating)
├── services/                 # New microservices
│   ├── model-gateway/
│   ├── rag-service/
│   ├── cascade-service/
│   ├── qa-service/
│   ├── agent-service/
│   ├── worker-service/
│   └── orchestration-service/
├── packages/                 # Shared libraries
│   └── dats-common/
├── deploy/                   # Deployment configs
│   ├── docker-compose.yml
│   └── k8s/
└── docs/                     # Architecture docs
```

## Rationale

### Why Monorepo Initially

1. **Atomic Changes**: Can update contracts and consumers in one commit
2. **Shared Tooling**: CI/CD, linting, testing infrastructure shared
3. **Refactoring**: Easier to move code between services during extraction
4. **Visibility**: All team members see all changes

### When to Split

Consider splitting when:
- Team size > 10 developers
- Services have independent release cycles
- Build times become prohibitive (> 10 minutes)
- Teams want different branching strategies

## Self-Contained Service Folder Requirements

Each service folder MUST be structured for independent deployment and future repo separation:

### Required Structure

```
services/{service-name}/
├── Dockerfile                 # Self-contained build
├── docker-compose.yml         # Local dev with dependencies
├── pyproject.toml             # Own dependencies (references dats-common)
├── requirements.txt           # Locked deps for reproducible builds
├── src/                       # Service code
├── tests/                     # Service tests (unit + integration)
├── config/                    # Service-specific config
├── contracts/                 # Service's OpenAPI/AsyncAPI specs
│   ├── openapi.yaml
│   └── asyncapi.yaml
├── Makefile                   # Common commands (build, test, lint)
└── README.md                  # Setup instructions
```

### Independence Rules

1. **No cross-service imports** - Services communicate only via HTTP or events
2. **Own CI pipeline** - Each service can be built/tested/deployed independently
3. **Self-sufficient tests** - Tests run without other services (use mocks/stubs)
4. **Health-first startup** - Service starts and passes health check without dependencies

### Validation Checklist

Before marking a service as "repo-ready":

- [ ] `docker build .` works from service folder
- [ ] `make test` passes without other services running
- [ ] `make lint` passes
- [ ] README contains complete setup instructions
- [ ] No imports from `../` or other service folders
- [ ] Contract files exist and are valid

---

## Consequences

### Positive
- Simplified initial development
- Easy cross-service refactoring
- Single CI/CD pipeline to maintain
- **Services ready for repo split when needed**
- **Teams can deploy independently (20+ deploys/day possible)**

### Negative
- All services versioned together
- Larger clone/build times as project grows
- Must be disciplined about service boundaries
- **Additional overhead maintaining per-service structure**

---

## Related Decisions

- ADR-002: Message Bus Selection
- ADR-003: API Format Selection