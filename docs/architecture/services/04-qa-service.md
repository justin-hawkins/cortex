# File: docs/architecture/services/04-qa-service.md
# QA Service

> **Priority**: P2 | **Team**: Quality | **Status**: Planned
>
> Output validation, automated checks, and human review management.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/qa-service/contracts/openapi.yaml` |
| AsyncAPI | `services/qa-service/contracts/asyncapi.yaml` |

---

## Purpose

- **Automated Validation**: Run validators (consensus, adversarial, security)
- **Human Review**: Queue outputs for human approval when needed
- **Profile Management**: Define validation profiles for different task types
- **Quality Metrics**: Track validation pass rates and issues

---

## API Endpoints

### POST /validate

Validate a task output.

```json
// Request
{"task_id": "task-123", "artifact_id": "art-456", "output": {"content": "def fibonacci..."}, "profile": "consensus"}

// Response
{"validation_id": "val-123", "status": "passed", "score": 0.92, "validators": [...], "requires_human_review": false}
```

### GET /reviews

List pending human reviews.

### POST /reviews/{review_id}/approve

Approve an output with optional comments.

### POST /reviews/{review_id}/reject

Reject an output with reason and issues.

### GET /profiles

List available QA profiles.

---

## Events

| Subscribes | Source | Action |
|------------|--------|--------|
| `task.output.created` | Worker Service | Queue for validation |
| `artifact.suspect` | Cascade Service | Re-validate artifact |

| Publishes | Trigger |
|-----------|---------|
| `task.validated` | Validation passed |
| `task.rejected` | Validation failed |
| `review.requested` | Human review needed |

---

## Configuration

```yaml
# config/qa-service.yaml
profiles:
  consensus:
    validators: [syntax, security, consensus]
    pass_threshold: 0.7
    human_review_threshold: 0.5
  strict:
    validators: [syntax, security, testing, adversarial]
    pass_threshold: 0.85

human_review:
  enabled: true
  sla_hours: 24

model_gateway:
  url: http://model-gateway:8000/api/v1
```

---

## Validators

| Validator | Purpose |
|-----------|---------|
| `syntax` | Code parsing, AST validation |
| `security` | Vulnerability scanning |
| `consensus` | Multi-model agreement (via Model Gateway) |
| `adversarial` | Adversarial review attempt |
| `testing` | Test execution and coverage |
| `documentation` | Doc quality checks |

---

## Migration Path

1. **Week 1**: Create service, move code from `src/qa/`
2. **Week 2**: Add event handlers, Model Gateway integration
3. Extract `validate_task` from `tasks.py`

---

## Environment Variables

```bash
MODEL_GATEWAY_URL=http://model-gateway:8000/api/v1
RABBITMQ_HOST=192.168.1.49
HUMAN_REVIEW_ENABLED=true
```

---

## Success Criteria

- [ ] Validation latency P95 < 30 seconds
- [ ] Human review SLA met 95% of time
- [ ] Zero false negatives on security issues
- [ ] Events published within 2 seconds
