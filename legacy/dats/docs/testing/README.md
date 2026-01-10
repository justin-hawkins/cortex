# Distributed Agentic Task System (DATS) - Testing Overview

## System Architecture Summary
```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                  │
│                    (CLI / API / Human Review Interface)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            COORDINATOR                                   │
│         Analyzes request, determines mode, identifies checkpoints        │
│                        Model: gpt-oss:120b                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            DECOMPOSER                                    │
│           Breaks tasks into subtasks until atomic                        │
│                        Model: gpt-oss:120b                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPLEXITY ESTIMATOR                               │
│         Routes tasks to appropriate tier based on complexity             │
│                        Model: gpt-oss:20b                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CELERY QUEUES                                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐                  │
│    │  tiny   │  │  small  │  │  large  │  │ frontier │                  │
│    │ gemma3  │  │ gemma3  │  │ gpt-oss │  │  claude  │                  │
│    │   4b    │  │   12b   │  │ 20b/120b│  │          │                  │
│    └─────────┘  └─────────┘  └─────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            WORKERS                                       │
│   Execute atomic tasks, can REFUSE if too complex                        │
│   Domains: code-general, code-vision, code-embedded, docs, ui            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          QA PIPELINE                                     │
│   Validates outputs: consensus, adversarial, security, human             │
│                     Model: gpt-oss:120b                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                    ┌──────────┐        ┌──────────┐
                    │   PASS   │        │   FAIL   │
                    └──────────┘        └──────────┘
                          │                   │
                          ▼                   ▼
                    ┌──────────┐        ┌──────────┐
                    │ Embedding│        │ Re-queue │
                    │ Trigger  │        │ or Escal │
                    └──────────┘        └──────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          LIGHTRAG                                        │
│              Project knowledge graph, embeddings                         │
│                    Model: mxbai-embed-large                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        SUPPORTING SYSTEMS                                │
├─────────────────────────────────────────────────────────────────────────┤
│  PROVENANCE         │  MERGE COORDINATOR    │  WORK PRODUCT STORE       │
│  Tracks lineage,    │  Resolves conflicts   │  GitHub repository        │
│  enables rollback   │  from parallel work   │                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Endpoints

| Component | Endpoint | Purpose |
|-----------|----------|---------|
| Ollama (small models) | http://192.168.1.12:11434 | gemma3:4b, gemma3:12b |
| Ollama (embedding) | http://192.168.1.11:11434 | mxbai-embed-large |
| vLLM (large models) | http://192.168.1.11:8000/v1 | gpt-oss:20b, gpt-oss:120b |
| Anthropic | https://api.anthropic.com/v1 | claude-sonnet-4-20250514 |
| Redis | localhost:6379 | Celery broker/backend |
| DATS API | localhost:8000 | System API |

## Data Flow Summary

1. **Request arrives** → Coordinator analyzes, sets mode
2. **Decomposition** → Task broken into subtasks with dependencies (DAG)
3. **Routing** → Each subtask assessed for complexity, routed to queue
4. **Execution** → Worker picks task, loads context from LightRAG, executes
5. **QA** → Output validated based on profile
6. **Storage** → Passed outputs saved to GitHub, embedded to LightRAG
7. **Provenance** → Full lineage recorded for traceability

## Test Documentation Index

1. [Component-Level Tests](./01-component-tests.md) - Individual component testing
2. [Integration Tests](./02-integration-tests.md) - Pipeline integration testing
3. [E2E Calculator Test](./03-e2e-calculator.md) - Full system validation
4. [Additional Test Programs](./04-additional-programs.md) - Complex scenario testing