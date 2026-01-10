# QA Service

> **DATS Microservice** - Output Validation & Review  
> Priority: P2  
> Team: Quality  
> Status: Planned

---

## Overview

### Purpose

The QA Service validates task outputs before they are marked as complete. It provides:

- **Automated Validation**: Run validators (consensus, adversarial, security, etc.)
- **Human Review**: Queue outputs for human approval when needed
- **Profile Management**: Define validation profiles for different task types
- **Quality Metrics**: Track validation pass rates and issues

### Current State (Monolith)

```
src/qa/
├── __init__.py
├── pipeline.py       # QAPipeline
├── profiles.py       # QA profile definitions
├── results.py        # Result dataclasses
├── human_review.py   # Human-in-the-loop
└── validators/
    ├── base.py
    ├── consensus.py
    ├── adversarial.py
    ├── security.py
    ├── testing.py
    └── documentation.py
```

**Problems:**
- `validate_task` in `tasks.py` couples QA to execution
- No independent scaling for validation
- Human review tightly integrated

---

## API Specification

### Base URL

```
http://qa-service:8000/api/v1
```

### Endpoints

#### POST /validate

Validate a task output.

**Request:**
```json
{
  "task_id": "task-123",
  "artifact_id": "art-456",
  "output": {
    "content": "def fibonacci(n)...",
    "artifacts": [{"type": "code", "language": "python"}]
  },
  "domain": "code-general",
  "profile": "consensus",
  "metadata": {
    "project_id": "proj-789"
  }
}
```

**Response:**
```json
{
  "validation_id": "val-123",
  "task_id": "task-123",
  "status": "passed",
  "score": 0.92,
  "profile": "consensus",
  "validators": [
    {
      "name": "syntax",
      "passed": true,
      "score": 1.0
    },
    {
      "name": "security",
      "passed": true,
      "score": 0.95,
      "issues": []
    },
    {
      "name": "consensus",
      "passed": true,
      "score": 0.85,
      "confidence": 0.9
    }
  ],
  "issues": [],
  "requires_human_review": false
}
```

#### GET /reviews

List pending human reviews.

**Response:**
```json
{
  "reviews": [
    {
      "review_id": "rev-123",
      "task_id": "task-456",
      "artifact_id": "art-789",
      "domain": "code-general",
      "reason": "low_confidence",
      "priority": "medium",
      "created_at": "2026-01-09T18:00:00Z"
    }
  ],
  "total": 5,
  "pending": 3
}
```

#### GET /reviews/{review_id}

Get review details with output content.

#### POST /reviews/{review_id}/approve

Approve an output.

**Request:**
```json
{
  "reviewer_id": "user-123",
  "comments": "Looks good, minor style issue fixed manually"
}
```

#### POST /reviews/{review_id}/reject

Reject an output.

**Request:**
```json
{
  "reviewer_id": "user-123",
  "reason": "logic_error",
  "comments": "The recursive call doesn't handle base case correctly",
  "issues": ["off_by_one", "missing_validation"]
}
```

#### GET /profiles

List available QA profiles.

#### GET /profiles/{name}

Get profile configuration.

#### GET /health

Health check.

---

## Events

### Subscribed Events

| Event | Source | Action |
|-------|--------|--------|
| `task.output.created` | Worker Service | Queue for validation |
| `artifact.suspect` | Cascade Service | Re-validate artifact |

### Published Events

| Event | Trigger | Data |
|-------|---------|------|
| `task.validated` | Validation passed | `{task_id, artifact_id, score}` |
| `task.rejected` | Validation failed | `{task_id, artifact_id, issues}` |
| `review.requested` | Human review needed | `{review_id, reason, priority}` |
| `review.completed` | Human decision made | `{review_id, decision, reviewer_id}` |

---

## Data Models

### ValidateRequest

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class OutputData(BaseModel):
    content: str
    artifacts: List[Dict[str, Any]]

class ValidateRequest(BaseModel):
    """Request to validate task output."""
    
    task_id: str
    artifact_id: str
    output: OutputData
    domain: str
    profile: str = "consensus"
    metadata: Optional[Dict[str, Any]] = None
```

### ValidationResult

```python
class ValidatorResult(BaseModel):
    name: str
    passed: bool
    score: float
    issues: List[str] = Field(default_factory=list)
    details: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    """Result of validation."""
    
    validation_id: str
    task_id: str
    status: str  # passed, failed, pending_review
    score: float
    profile: str
    validators: List[ValidatorResult]
    issues: List[str]
    requires_human_review: bool
```

---

## Architecture

### Internal Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       QA SERVICE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests                              │
│  │  Router     │◄─── RabbitMQ Events                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                   VALIDATION PIPELINE                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ Profile  │─▶│ Validator│─▶│  Result  │              │   │
│  │  │ Resolver │  │  Runner  │  │ Aggregator│              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      VALIDATORS                           │   │
│  │  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ Syntax │  │ Security │  │Consensus │  │Adversarial│  │   │
│  │  └────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  │  ┌────────┐  ┌──────────┐  ┌──────────┐               │   │
│  │  │Testing │  │  Docs    │  │  Custom  │               │   │
│  │  └────────┘  └──────────┘  └──────────┘               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  HUMAN REVIEW MANAGER                     │   │
│  │  - Queue management                                       │   │
│  │  - Reviewer assignment                                    │   │
│  │  - Decision tracking                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  MODEL GATEWAY CLIENT                     │   │
│  │  - For consensus/adversarial validation                   │   │
│  │  - LLM-based code review                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/qa-service.yaml
profiles:
  consensus:
    validators:
      - name: syntax
        weight: 0.3
      - name: security
        weight: 0.3
      - name: consensus
        weight: 0.4
        config:
          model_count: 3
          agreement_threshold: 0.66
    pass_threshold: 0.7
    human_review_threshold: 0.5
    
  strict:
    validators:
      - name: syntax
        weight: 0.2
      - name: security
        weight: 0.3
      - name: testing
        weight: 0.3
      - name: adversarial
        weight: 0.2
    pass_threshold: 0.85
    human_review_threshold: 0.7

human_review:
  enabled: true
  auto_assign: false
  sla_hours: 24
  escalation_hours: 48
  
model_gateway:
  url: http://model-gateway:8000/api/v1
  
events:
  rabbitmq:
    host: rabbitmq
    port: 5672
    exchange: task.events
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Observability

### Metrics

```
# Validation
qa_validations_total{profile="consensus", result="passed"}
qa_validation_score{profile="consensus", quantile="0.5"}
qa_validation_latency_seconds{profile="consensus", quantile="0.95"}

# Validators
qa_validator_passed_total{validator="security"}
qa_validator_issues_total{validator="security", issue_type="vulnerability"}

# Human review
qa_human_review_pending
qa_human_review_completed_total{decision="approved"}
qa_human_review_sla_exceeded_total
```

---

## Success Criteria

- [ ] Validation latency P95 < 30 seconds
- [ ] Pass rate tracking accurate to 1%
- [ ] Human review SLA met 95% of time
- [ ] Zero false negatives on security issues
- [ ] Events published within 2 seconds

---

*Last updated: January 2026*