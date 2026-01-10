# Agent Service

> **DATS Microservice** - Task Analysis & Decomposition  
> Priority: P3  
> Team: AI/ML  
> Status: Planned

---

## Overview

### Purpose

The Agent Service provides intelligent task analysis capabilities:

- **Coordinator**: Analyze requests, determine mode (new_project, modify, fix_bug)
- **Decomposer**: Break complex tasks into atomic subtasks
- **Complexity Estimator**: Route tasks to appropriate model tier

### Current State (Monolith)

```
src/agents/
├── __init__.py
├── base.py                # BaseAgent
├── coordinator.py         # Coordinator
├── decomposer.py          # Decomposer
├── complexity_estimator.py
├── qa_reviewer.py         # (→ QA Service)
└── merge_coordinator.py   # (→ Worker Service)
```

---

## API Specification

### Base URL

```
http://agent-service:8000/api/v1
```

### Endpoints

#### POST /analyze

Analyze a task request.

**Request:**
```json
{
  "request": "Create a Python function that calculates fibonacci numbers",
  "project_id": "proj-123",
  "context": {
    "existing_files": ["utils.py", "tests/"],
    "language": "python"
  }
}
```

**Response:**
```json
{
  "analysis_id": "ana-123",
  "mode": "new_project",
  "domain": "code-general",
  "needs_decomposition": false,
  "estimated_complexity": "small",
  "acceptance_criteria": "Function should handle edge cases...",
  "qa_profile": "consensus"
}
```

#### POST /decompose

Decompose a complex task.

**Request:**
```json
{
  "task_id": "task-123",
  "description": "Build a REST API for user management",
  "max_depth": 5
}
```

**Response:**
```json
{
  "parent_task_id": "task-123",
  "subtasks": [
    {
      "id": "sub-1",
      "description": "Create User model with validation",
      "domain": "code-general",
      "is_atomic": true,
      "dependencies": []
    },
    {
      "id": "sub-2",
      "description": "Implement CRUD endpoints",
      "domain": "code-general",
      "is_atomic": true,
      "dependencies": ["sub-1"]
    }
  ],
  "decomposition_depth": 1
}
```

#### POST /estimate

Estimate task complexity and route to tier.

**Request:**
```json
{
  "task_id": "task-123",
  "description": "Create fibonacci function",
  "domain": "code-general"
}
```

**Response:**
```json
{
  "task_id": "task-123",
  "recommended_tier": "small",
  "estimated_tokens": 500,
  "confidence": 0.85,
  "reasoning": "Simple algorithmic task, limited context needed",
  "qa_profile": "consensus"
}
```

#### GET /health

Health check.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT SERVICE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests                              │
│  │  Router     │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                      AGENTS                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │   │
│  │  │Coordinator │  │ Decomposer │  │ComplexityEstimator│   │   │
│  │  │            │  │            │  │                  │   │   │
│  │  │ - Mode     │  │ - Recursive│  │ - Token estimate │   │   │
│  │  │ - Domain   │  │ - Atomic   │  │ - Tier routing   │   │   │
│  │  │ - QA prof  │  │ - DAG build│  │ - QA profile     │   │   │
│  │  └────────────┘  └────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  MODEL GATEWAY CLIENT                     │   │
│  │  - All LLM calls go through gateway                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  PROMPT RENDERER                          │   │
│  │  - Jinja2 templates from prompts/ directory              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/agent-service.yaml
agents:
  coordinator:
    preferred_tier: large
    timeout_seconds: 60
    
  decomposer:
    preferred_tier: large
    max_depth: 5
    max_subtasks: 20
    
  complexity_estimator:
    preferred_tier: small
    timeout_seconds: 30

prompts:
  templates_dir: /app/prompts
  
model_gateway:
  url: http://model-gateway:8000/api/v1
```

---

## Migration Path

1. Create `services/agent-service/` directory
2. Move agents from `src/agents/`:
   - `coordinator.py`
   - `decomposer.py`
   - `complexity_estimator.py`
   - `base.py`
3. Update to use Model Gateway client
4. Copy prompt templates to service

---

## Success Criteria

- [ ] Analysis latency P95 < 20 seconds
- [ ] Decomposition produces valid DAG
- [ ] Tier routing accuracy > 90%
- [ ] Stateless and horizontally scalable

---

*Last updated: January 2026*