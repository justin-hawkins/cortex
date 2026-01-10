# Cortex

A distributed AI agent orchestration platform for complex software development tasks.

## Overview

Cortex is a system for orchestrating multiple AI agents to collaboratively decompose, execute, and validate software development tasks. It uses a hierarchical architecture with specialized workers, distributed task queues, and comprehensive quality assurance.

## Project Structure

```
cortex/
├── dats/                    # Distributed Agentic Task System (core)
│   ├── src/                 # Source code
│   │   ├── agents/          # Orchestration agents
│   │   ├── workers/         # Specialized task executors
│   │   ├── queue/           # Celery task distribution
│   │   ├── qa/              # Quality assurance pipeline
│   │   ├── merge/           # Conflict resolution
│   │   ├── rag/             # LightRAG integration
│   │   └── api/             # REST API
│   ├── tests/               # Test suite
│   └── docs/testing/        # Testing documentation
├── prompts/                 # Prompt templates and schemas
│   ├── agents/              # Agent system prompts
│   ├── workers/             # Worker execution prompts
│   └── schemas/             # YAML schemas
└── celery-unraid/           # Docker deployment for workers
```

## Components

### DATS (Distributed Agentic Task System)
The core orchestration system. See [dats/README.md](dats/README.md) for detailed documentation.

**Key Features:**
- Hierarchical agent architecture (Coordinator → Decomposer → Workers)
- Model-tier-based task routing (tiny/small/large/frontier)
- Quality assurance with consensus validation
- Provenance tracking and cascade failure handling
- LightRAG integration for project knowledge

### Prompts
Template library for agent and worker prompts. See [prompts/README.md](prompts/README.md) for details.

**Contents:**
- Agent prompts: coordinator, decomposer, complexity estimator, QA reviewer
- Worker prompts: code_general, code_vision, code_embedded, documentation, ui_design
- Schemas: task definitions, provenance tracking, routing configuration

### Celery Workers (Docker)
Containerized Celery workers for distributed deployment. Located in `celery-unraid/`.

## Quick Start

### Prerequisites
- Python 3.10+
- Redis (for Celery results backend)
- RabbitMQ (for Celery message broker)
- Access to Ollama/vLLM servers or Anthropic API

### Installation

```bash
# Clone the repository
git clone https://github.com/justin-hawkins/cortex.git
cd cortex

# Set up DATS
cd dats
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Configure environment
cp .env.example .env  # Edit with your settings
```

### Running

```bash
# Start Celery workers
celery -A src.queue.celery_app worker -Q tiny,small,large,frontier,default -l INFO

# Submit a task
dats submit --project my-project --mode autonomous "Create a hello world Python script"

# Check status
dats status <task-id>
```

## Server Infrastructure

| Server | IP | Purpose |
|--------|-----|---------|
| epyc_server | 192.168.1.11 | vLLM (large models), Ollama (CPU) |
| rtx4060_server | 192.168.1.12 | Ollama (GPU, small models) |
| RabbitMQ | 192.168.1.49:5672 | Message broker |
| Redis | 192.168.1.44:6379 | Results backend |

## Documentation

| Document | Description |
|----------|-------------|
| [dats/README.md](dats/README.md) | DATS architecture, installation, and usage |
| [prompts/README.md](prompts/README.md) | Prompt template system |
| [dats/docs/testing/](dats/docs/testing/) | Comprehensive testing documentation |

### Testing Documentation
- [Testing Overview](dats/docs/testing/README.md) - System architecture and component endpoints
- [Component Tests](dats/docs/testing/01-component-tests.md) - Unit tests for each component
- [Integration Tests](dats/docs/testing/02-integration-tests.md) - Pipeline integration tests
- [E2E Calculator Test](dats/docs/testing/03-e2e-calculator.md) - Full system validation
- [Additional Test Programs](dats/docs/testing/04-additional-programs.md) - Complex test scenarios

## Architecture

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
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            DECOMPOSER                                    │
│           Breaks tasks into subtasks until atomic                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPLEXITY ESTIMATOR                               │
│         Routes tasks to appropriate tier based on complexity             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CELERY QUEUES                                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐                  │
│    │  tiny   │  │  small  │  │  large  │  │ frontier │                  │
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
└─────────────────────────────────────────────────────────────────────────┘
```

## License

MIT License