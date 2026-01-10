# File: docs/architecture/services/05-agent-service.md
# Agent Service

> **Priority**: P1 | **Team**: AI/ML | **Status**: Planned
>
> Task analysis, decomposition, and complexity estimation.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/agent-service/contracts/openapi.yaml` |
| AsyncAPI | N/A (stateless, uses Model Gateway) |

---

## Purpose

- **Coordinator**: Analyze requests, determine mode (new_project, modify, fix_bug)
- **Decomposer**: Break complex tasks into atomic subtasks
- **Complexity Estimator**: Route tasks to appropriate model tier

---

## API Endpoints

### POST /analyze

Analyze a task request.

```json
// Request
{"request": "Create a Python fibonacci function", "project_id": "proj-123"}

// Response
{"mode": "new_project", "domain": "code-general", "needs_decomposition": false, "estimated_complexity": "small", "qa_profile": "consensus"}
```

### POST /decompose

Decompose a complex task into subtasks.

```json
// Request
{"task_id": "task-123", "description": "Build a REST API for user management", "max_depth": 5}

// Response
{"subtasks": [{"id": "sub-1", "description": "Create User model", "is_atomic": true, "dependencies": []}], "decomposition_depth": 1}
```

### POST /estimate

Estimate task complexity and route to tier.

```json
// Request
{"task_id": "task-123", "description": "Create fibonacci function", "domain": "code-general"}

// Response
{"recommended_tier": "small", "estimated_tokens": 500, "confidence": 0.85, "qa_profile": "consensus"}
```

---

## Configuration

```yaml
# config/agent-service.yaml
agents:
  coordinator:
    preferred_tier: large  # openai/gpt-oss-20b
  decomposer:
    preferred_tier: large
    max_depth: 5
    max_subtasks: 20
  complexity_estimator:
    preferred_tier: small  # gemma3:12b

prompts:
  templates_dir: /app/prompts

model_gateway:
  url: http://model-gateway:8000/api/v1
```

---

## Migration Path

1. **Week 1**: Create service, move agents from `src/agents/`
2. **Week 2**: Update to use Model Gateway client
3. Copy prompt templates to service

---

## Environment Variables

```bash
MODEL_GATEWAY_URL=http://model-gateway:8000/api/v1
PROMPTS_DIR=/app/prompts
```

---

## Success Criteria

- [ ] Analysis latency P95 < 20 seconds
- [ ] Decomposition produces valid DAG
- [ ] Tier routing accuracy > 90%
- [ ] Stateless and horizontally scalable
