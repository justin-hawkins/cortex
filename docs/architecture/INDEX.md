# File: docs/architecture/INDEX.md
# DATS Architecture Documentation

> **Distributed Agentic Task System** - Documentation Hub
> 
> Start here to navigate the architecture documentation.

---

## 📋 Getting Started

| Document | Description |
|----------|-------------|
| [Microservices Design](MICROSERVICES_DESIGN.md) | Overall architecture, service inventory, communication patterns |
| [Service Definition of Done](SERVICE_DONE_DEFINITION.md) | What "done" means for service changes |
| [servers.yaml](servers.yaml) | Infrastructure endpoints (Ollama, vLLM, RabbitMQ, Redis) |

---

## 🏗️ Services

| Service | Document | Priority | Team | Status |
|---------|----------|----------|------|--------|
| Model Gateway | [01-model-gateway.md](services/01-model-gateway.md) | P0 | Platform | Planned |
| RAG Service | [02-rag-service.md](services/02-rag-service.md) | P-DEFERRED | Data | Deferred |
| Cascade Service | [03-cascade-service.md](services/03-cascade-service.md) | P3 | Reliability | Planned |
| QA Service | [04-qa-service.md](services/04-qa-service.md) | P3 | Quality | Planned |
| Agent Service | [05-agent-service.md](services/05-agent-service.md) | P1 | AI/ML | Planned |
| Worker Service | [06-worker-service.md](services/06-worker-service.md) | P1 | AI/ML | Planned |
| Orchestration Service | [07-orchestration-service.md](services/07-orchestration-service.md) | P2 | Platform | Planned |

---

## 📜 Architecture Decision Records (ADRs)

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](decisions/001-repo-strategy.md) | Repository Strategy (Monorepo → Multi-repo) | Accepted |
| [ADR-002](decisions/002-message-bus.md) | Message Bus Selection (RabbitMQ) | Planned |
| [ADR-003](decisions/003-api-format.md) | API Format (REST + gRPC-ready) | Planned |
| [ADR-004](decisions/004-service-boundaries.md) | Service Boundaries | Planned |
| [ADR-005](decisions/005-contract-strategy.md) | Service Contract Strategy (OpenAPI + AsyncAPI) | Accepted |

---

## 📄 Contracts

| Resource | Description |
|----------|-------------|
| [Contract Guidelines](contracts/README.md) | How to define and use service contracts |
| `contracts/openapi/` | OpenAPI specs for REST endpoints (coming soon) |
| `contracts/asyncapi/` | AsyncAPI specs for event contracts (coming soon) |
| `contracts/schemas/` | Shared JSON schemas (coming soon) |

---

## 🔧 Shared Resources

| Resource | Description |
|----------|-------------|
| [Prompts README](prompts/README.md) | Agent and worker prompt templates |
| [prompts/schemas/](prompts/schemas/) | Task, provenance, and routing schemas |
| `packages/dats-common/` | Shared Python package (to be created) |

---

## 🗺️ Document Map

```
docs/architecture/
├── INDEX.md                          ← You are here
├── MICROSERVICES_DESIGN.md           # Main architecture document
├── SERVICE_DONE_DEFINITION.md        # Definition of done for services
├── servers.yaml                      # Infrastructure endpoints
│
├── decisions/                        # Architecture Decision Records
│   ├── 001-repo-strategy.md
│   ├── 005-contract-strategy.md
│   └── ...
│
├── services/                         # Individual service specifications
│   ├── 01-model-gateway.md
│   ├── 02-rag-service.md
│   ├── 03-cascade-service.md
│   ├── 04-qa-service.md
│   ├── 05-agent-service.md
│   ├── 06-worker-service.md
│   └── 07-orchestration-service.md
│
├── contracts/                        # Service contracts
│   ├── README.md
│   ├── openapi/                      # REST API specs
│   ├── asyncapi/                     # Event specs
│   └── schemas/                      # Shared data schemas
│
└── prompts/                          # LLM prompt templates
    ├── agents/
    ├── workers/
    └── schemas/
```

---

## 🚀 Quick Links by Role

### For New Team Members
1. Read [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md) for architecture overview
2. Review [ADR-001](decisions/001-repo-strategy.md) for repo structure
3. Check your team's service document in `services/`

### For Developers
1. Review [Service Definition of Done](SERVICE_DONE_DEFINITION.md) before submitting PRs
2. Check [Contract Guidelines](contracts/README.md) for API contract requirements
3. Reference [servers.yaml](servers.yaml) for infrastructure endpoints

### For Architects
1. ADRs in `decisions/` for architectural decisions
2. Service docs in `services/` for detailed designs
3. [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md) for the big picture

---

*Last updated: January 2026*