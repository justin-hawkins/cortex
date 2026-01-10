# File: docs/architecture/SERVICE_DONE_DEFINITION.md
# Service Definition of Done

> **DATS Microservices** - What "Done" Means for Service Changes
> 
> This document defines the requirements that must be met before any service change 
> can be considered complete. Similar to project-level `.clinerules`, this ensures
> consistency and quality across all teams.

---

## Quick Reference

All service changes require:

| Category | Requirement | Enforced By |
|----------|-------------|-------------|
| Contracts | OpenAPI/AsyncAPI updated | CI validation |
| Testing | Unit + contract tests | CI pipeline |
| Documentation | README current | PR review |
| Deployment | Health check works | CI + deploy |

---

## Required for ALL Service Changes

### ✅ Contract Compliance

Every change that affects service interfaces must update contracts:

- [ ] **OpenAPI spec updated** if REST endpoints changed (added/modified/removed)
- [ ] **AsyncAPI spec updated** if events changed (published/subscribed)
- [ ] **Contract version bumped** if breaking change (see versioning rules below)
- [ ] **Contract tests pass** validating implementation matches spec
- [ ] **No breaking changes** unless explicitly approved with migration plan

#### Contract Versioning Rules

| Change Type | Action Required |
|-------------|-----------------|
| New optional field | MINOR version bump (1.0.0 → 1.1.0) |
| New endpoint | MINOR version bump |
| Removed field | MAJOR version bump + migration plan |
| Changed field type | MAJOR version bump + migration plan |
| Documentation fix | PATCH version bump (1.0.0 → 1.0.1) |

---

### ✅ Testing Requirements

All changes must include appropriate tests:

- [ ] **Unit tests** for new/changed code (minimum 80% coverage on new code)
- [ ] **Integration tests** for service dependencies (use mocks for external services)
- [ ] **Contract tests** for API/event consumers (Pact or similar)
- [ ] **All tests pass locally** before PR (`make test`)
- [ ] **CI pipeline green** before merge

#### Test Categories

```
tests/
├── unit/           # Fast, isolated, mock all dependencies
├── integration/    # Test with real DB, mocked external services  
├── contract/       # Validate against OpenAPI/AsyncAPI specs
└── e2e/            # Full flow tests (run in staging only)
```

---

### ✅ Documentation Requirements

Documentation must stay current:

- [ ] **README updated** if setup/config changed
- [ ] **API docs regenerated** from OpenAPI (if applicable)
- [ ] **CHANGELOG entry added** for user-visible changes
- [ ] **Architecture docs updated** if design changed (in `docs/architecture/`)
- [ ] **Inline code comments** for complex logic

#### CHANGELOG Format

```markdown
## [1.2.0] - 2026-01-15

### Added
- New `/health/detailed` endpoint for deep health checks

### Changed  
- Increased default timeout from 30s to 60s

### Fixed
- Race condition in request handling
```

---

### ✅ Deployment Readiness

Service must be independently deployable:

- [ ] **Dockerfile builds** successfully (`docker build .`)
- [ ] **Health check endpoint** works (`/health` or `/api/v1/health`)
- [ ] **Service starts independently** without other services running
- [ ] **No cross-service imports** (only via HTTP/events)
- [ ] **Environment variables documented** in README
- [ ] **Graceful shutdown** handles SIGTERM properly

#### Health Check Requirements

```python
# Minimum health check response
{
    "status": "healthy",  # or "unhealthy"
    "version": "1.2.0",
    "timestamp": "2026-01-15T10:30:00Z"
}

# Detailed health check (optional /health/detailed)
{
    "status": "healthy",
    "version": "1.2.0",
    "dependencies": {
        "database": "connected",
        "model_gateway": "connected",
        "rabbitmq": "connected"
    }
}
```

---

### ✅ Backward Compatibility

Changes must not break consumers:

- [ ] **API backward compatible** (old requests still work)
- [ ] **Event schema backward compatible** (old consumers can process new events)
- [ ] **Migration path documented** if breaking change unavoidable
- [ ] **Deprecation warnings** added for fields to be removed
- [ ] **Feature flags** used for gradual rollout (when appropriate)

#### Backward Compatibility Checklist

| Before | After | Compatible? |
|--------|-------|-------------|
| Required field | Optional field | ✅ Yes |
| Optional field | Required field | ❌ No (breaking) |
| Add new optional field | - | ✅ Yes |
| Remove field | - | ❌ No (breaking) |
| Change field type | - | ❌ No (breaking) |
| Add new endpoint | - | ✅ Yes |
| Remove endpoint | - | ❌ No (breaking) |

---

## PR Checklist

Copy this into your PR description:

```markdown
## Definition of Done Checklist

### Contracts
- [ ] OpenAPI/AsyncAPI specs updated (if API changed)
- [ ] Contract version bumped appropriately
- [ ] Contract tests pass

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] All CI checks green

### Documentation
- [ ] README updated (if needed)
- [ ] CHANGELOG entry added

### Deployment
- [ ] Dockerfile builds successfully
- [ ] Health check works
- [ ] No cross-service imports
```

---

## Exceptions

### When Can Requirements Be Skipped?

| Scenario | Allowed Exceptions |
|----------|-------------------|
| Hotfix (P0 incident) | Skip CHANGELOG, update within 24h |
| Internal refactor (no API change) | Skip contract updates |
| Documentation-only change | Skip tests, deployment checks |
| Prototype/experiment | Skip all (must be on feature branch) |

All exceptions must be noted in PR description with justification.

---

## Enforcement

### CI Pipeline Checks

The following are automatically enforced:

```yaml
# Required CI checks
- contract-validation     # OpenAPI/AsyncAPI linting
- unit-tests             # pytest with coverage
- integration-tests      # Service integration tests
- contract-tests         # Pact or schema validation
- docker-build           # Dockerfile builds
- health-check           # Service starts and responds
```

### PR Review Requirements

Reviewers should verify:

1. All checklist items completed
2. Tests are meaningful (not just coverage padding)
3. Documentation matches implementation
4. No breaking changes without approval

---

## Related Documents

- [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md) - Overall architecture
- [ADR-005: Contract Strategy](decisions/005-contract-strategy.md) - Contract decisions
- [ADR-001: Repository Strategy](decisions/001-repo-strategy.md) - Folder structure
- [Contract Guidelines](contracts/README.md) - How to write contracts

---

*This document is enforced by CI/CD pipelines and PR review. Updates require architecture team approval.*