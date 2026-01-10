# File: docs/architecture/services/03-cascade-service.md
# Cascade Service

> **Priority**: P2 | **Team**: Reliability | **Status**: Planned
>
> Failure propagation, taint tracking, revalidation, and rollback.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/cascade-service/contracts/openapi.yaml` |
| AsyncAPI | `services/cascade-service/contracts/asyncapi.yaml` |

---

## Purpose

- **Taint Propagation**: Mark artifacts and dependents as tainted/suspect
- **Revalidation**: Queue suspect artifacts for re-verification
- **Rollback**: Restore to previous checkpoints
- **Impact Analysis**: Calculate blast radius of failures

---

## API Endpoints

### POST /taint

Initiate taint propagation for an artifact.

```json
// Request
{"artifact_id": "art-123", "reason": "qa_failure", "severity": "high"}

// Response
{"cascade_id": "cascade-789", "tainted_count": 1, "suspect_count": 5, "revalidation_queued": 5}
```

### GET /impact/{artifact_id}

Analyze impact if artifact were tainted. Returns direct/transitive dependents.

### POST /rollback

Rollback to a checkpoint.

```json
// Request
{"checkpoint_id": "chk-123", "reason": "cascade_threshold_exceeded", "dry_run": false}

// Response
{"rollback_id": "rb-456", "artifacts_reverted": 15, "tasks_to_requeue": 8}
```

### POST /checkpoints

Create a checkpoint for a project.

### GET /revalidation/queue

Get revalidation queue status.

---

## Events

| Subscribes | Source | Action |
|------------|--------|--------|
| `task.rejected` | QA Service | Detect cascade, propagate taint |
| `security.alert` | External | Security-triggered cascade |

| Publishes | Trigger |
|-----------|---------|
| `cascade.started` | Taint initiated |
| `artifact.tainted` | After taint |
| `artifact.suspect` | Dependent flagged |
| `revalidation.needed` | Suspect queued |
| `rollback.requested` | Threshold exceeded |

---

## Configuration

```yaml
# config/cascade-service.yaml
cascade:
  max_depth: 10
  batch_size: 50

thresholds:
  suspect_count: 100
  cascade_depth: 5
  project_impact_ratio: 0.3

revalidation:
  queue_max_size: 1000
  retry_attempts: 3

rollback:
  require_approval: true
  max_rollback_age_days: 30
```

---

## Migration Path

1. **Week 1**: Create service, move code from `src/cascade/`
2. **Week 2**: Add provenance client, event handlers
3. Extract cascade tasks from `tasks.py`: `propagate_taint`, `queue_revalidation`, `execute_rollback`

---

## Environment Variables

```bash
RABBITMQ_HOST=192.168.1.49
REDIS_URL=redis://192.168.1.44:6379/0
PROVENANCE_SERVICE_URL=http://orchestration-service:8000/api/v1
```

---

## Success Criteria

- [ ] Taint propagation < 30 seconds for 100 artifacts
- [ ] Revalidation queue processes 50 items/minute
- [ ] Zero missed dependents in propagation
- [ ] Rollback restores to checkpoint correctly
