# File: docs/architecture/services/06-worker-service.md
# Worker Service

> **Priority**: P1 | **Team**: AI/ML | **Status**: Planned
>
> Code generation execution via domain-specific workers.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/worker-service/contracts/openapi.yaml` |
| AsyncAPI | `services/worker-service/contracts/asyncapi.yaml` |

---

## Purpose

- **Domain-Specific Workers**: code-general, code-vision, code-embedded, documentation, ui-design
- **Context Integration**: Receives context from RAG Service
- **Output Production**: Generates artifacts for QA validation

---

## API Endpoints

### POST /execute

Execute a task (synchronous for small tasks).

```json
// Request
{"task_id": "task-123", "description": "Create fibonacci function", "domain": "code-general", "tier": "small", "context": "..."}

// Response
{"task_id": "task-123", "artifact_id": "art-789", "status": "completed", "output": {"content": "def fibonacci..."}, "tokens_used": 450}
```

---

## Events

| Subscribes | Source | Action |
|------------|--------|--------|
| `task.ready.{tier}` | Orchestration | Execute task |

| Publishes | Trigger |
|-----------|---------|
| `task.output.created` | Generation complete |
| `task.failed` | Generation failed |

---

## Domain Workers

| Worker | Description | Default Tier |
|--------|-------------|--------------|
| `code-general` | Standard code generation | small |
| `code-vision` | Image/visual code | large |
| `code-embedded` | Embedded systems | small |
| `documentation` | Doc generation | small |
| `ui-design` | UI components | large |

---

## Configuration

```yaml
# config/worker-service.yaml
workers:
  code-general:
    prompt_template: workers/code_general.md
    default_tier: small
  code-vision:
    prompt_template: workers/code_vision.md
    default_tier: large

tier_models:
  tiny: {model: gemma3:4b, endpoint: ollama_gpu_general}
  small: {model: gemma3:12b, endpoint: ollama_gpu_general}
  large: {model: openai/gpt-oss-20b, endpoint: vllm_gpu}
  coding: {model: qwen3-coder:30b-a3b-q8_0-64k, endpoint: ollama_cpu_large}
  frontier: {model: claude-sonnet-4-20250514, endpoint: anthropic}

queues:
  tiny: {concurrency: 10, timeout: 60}
  small: {concurrency: 5, timeout: 120}
  large: {concurrency: 2, timeout: 300}
  frontier: {concurrency: 1, timeout: 600}
```

---

## Migration Path

1. **Week 1**: Create service, move workers from `src/workers/`
2. **Week 2**: Add event handlers, Model Gateway + RAG integration
3. Extract worker execution from `tasks.py`

---

## Environment Variables

```bash
MODEL_GATEWAY_URL=http://model-gateway:8000/api/v1
RAG_SERVICE_URL=http://rag-service:8000/api/v1
RABBITMQ_HOST=192.168.1.49
PROMPTS_DIR=/app/prompts
```

---

## Success Criteria

- [ ] P95 execution time within tier SLA
- [ ] Zero lost tasks from queue
- [ ] Output validates against schema
- [ ] RAG context integration working
