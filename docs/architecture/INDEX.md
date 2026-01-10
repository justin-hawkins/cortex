# File: docs/architecture/INDEX.md
# DATS Architecture Documentation

> **Distributed Agentic Task System** - Documentation Hub

---

## 📋 Getting Started

| Document | Description |
|----------|-------------|
| [Microservices Design](MICROSERVICES_DESIGN.md) | Overall architecture, communication patterns |
| [Service Common Patterns](_shared/SERVICE_COMMON.md) | Shared boilerplate for all services |
| [Service Definition of Done](SERVICE_DONE_DEFINITION.md) | PR requirements for service changes |
| [servers.yaml](servers.yaml) | Infrastructure endpoints |

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

## 🗺️ Document Structure

```
docs/architecture/
├── INDEX.md                     ← You are here
├── MICROSERVICES_DESIGN.md      # Main architecture
├── SERVICE_DONE_DEFINITION.md   # PR requirements
├── servers.yaml                 # Infrastructure endpoints
├── _shared/
│   └── SERVICE_COMMON.md        # Shared service patterns
├── decisions/                   # ADRs
├── services/                    # Per-service specs (reference _shared/)
├── contracts/                   # OpenAPI/AsyncAPI specs
└── prompts/                     # LLM prompt templates
```

---

## 🚀 Quick Links

| Role | Start Here |
|------|------------|
| New Team Member | [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md), then your team's service doc |
| Developer | [SERVICE_DONE_DEFINITION.md](SERVICE_DONE_DEFINITION.md), [Contract Guidelines](contracts/README.md) |
| Architect | [decisions/](decisions/) for ADRs, [MICROSERVICES_DESIGN.md](MICROSERVICES_DESIGN.md) |

---

*Last updated: January 2026*
