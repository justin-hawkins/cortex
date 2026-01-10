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

## Consequences

### Positive
- Simplified initial development
- Easy cross-service refactoring
- Single CI/CD pipeline to maintain

### Negative
- All services versioned together
- Larger clone/build times as project grows
- Must be disciplined about service boundaries

---

## Related Decisions

- ADR-002: Message Bus Selection
- ADR-003: API Format Selection