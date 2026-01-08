# DATS - Distributed Agentic Task System

A distributed system for orchestrating AI agents to decompose, execute, and validate complex software development tasks.

## Overview

DATS uses a hierarchical agent architecture with specialized workers to handle different types of tasks. Tasks are distributed via Celery queues with model-tier-based routing.

## Architecture

```
┌─────────────────┐
│   Coordinator   │  ← Entry point for high-level tasks
└────────┬────────┘
         │
    ┌────▼────┐
    │Decomposer│  ← Breaks down complex tasks
    └────┬────┘
         │
┌────────▼────────┐
│Complexity       │  ← Estimates task requirements
│Estimator        │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Queue   │  ← Celery task routing
    └────┬────┘
         │
┌────────▼────────┐
│    Workers      │  ← Specialized executors
│ ├─ code_general │
│ ├─ code_vision  │
│ ├─ code_embedded│
│ ├─ documentation│
│ └─ ui_design    │
└────────┬────────┘
         │
┌────────▼────────┐
│  QA Reviewer    │  ← Quality validation
└────────┬────────┘
         │
┌────────▼────────┐
│Merge Coordinator│  ← Combines subtask results
└─────────────────┘
```

## Model Tiers

| Tier     | Models                          | Context | Use Case                    |
|----------|--------------------------------|---------|------------------------------|
| tiny     | gemma3:4b                      | 32k     | Simple transformations       |
| small    | gemma3:12b                     | 32k     | Standard coding tasks        |
| large    | qwen3-coder, gpt-oss:20b       | 64k     | Complex reasoning            |
| frontier | claude-sonnet-4                | 200k    | Most complex tasks           |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Configuration

Create a `.env` file with your configuration:

```env
# RabbitMQ
RABBITMQ_HOST=192.168.1.49
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Redis (for results)
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys
ANTHROPIC_API_KEY=your-key-here
GITHUB_TOKEN=your-token-here

# Paths
PROMPTS_DIR=../prompts
ROUTING_CONFIG_PATH=../prompts/schemas/routing_config.yaml
```

## Usage

### Starting Workers

Start Celery workers for each tier:

```bash
# Start all workers
celery -A src.queue.celery_app worker -Q tiny,small,large,frontier,default -l INFO

# Or start tier-specific workers
celery -A src.queue.celery_app worker -Q tiny -l INFO --hostname=tiny@%h
celery -A src.queue.celery_app worker -Q small -l INFO --hostname=small@%h
celery -A src.queue.celery_app worker -Q large -l INFO --hostname=large@%h
celery -A src.queue.celery_app worker -Q frontier -l INFO --hostname=frontier@%h
```

### Submitting Tasks

```python
from src.queue.tasks import execute_task

# Submit a task
result = execute_task.delay({
    "id": "task-123",
    "project_id": "project-456",
    "type": "execute",
    "domain": "code-general",
    "description": "Implement a hello world function",
    "routing": {"tier": "small"},
})

# Wait for result
output = result.get(timeout=300)
```

### Using Agents

```python
import asyncio
from src.agents.coordinator import Coordinator

async def main():
    coordinator = Coordinator()
    
    result = await coordinator.analyze_task({
        "id": "task-123",
        "project_id": "project-456",
        "description": "Build a REST API for user management",
    })
    
    print(result)

asyncio.run(main())
```

## Project Structure

```
dats/
├── src/
│   ├── config/          # Configuration management
│   │   ├── settings.py  # Environment variables
│   │   └── routing.py   # Model routing config
│   ├── models/          # Model client abstractions
│   │   ├── base.py      # Abstract base class
│   │   ├── ollama_client.py
│   │   ├── openai_client.py
│   │   └── anthropic_client.py
│   ├── prompts/         # Prompt management
│   │   ├── loader.py    # Template loading
│   │   └── renderer.py  # Variable substitution
│   ├── queue/           # Celery infrastructure
│   │   ├── celery_app.py
│   │   └── tasks.py
│   ├── agents/          # Orchestration agents
│   │   ├── coordinator.py
│   │   ├── decomposer.py
│   │   ├── complexity_estimator.py
│   │   ├── qa_reviewer.py
│   │   └── merge_coordinator.py
│   ├── workers/         # Task execution workers
│   │   ├── code_general.py
│   │   ├── code_vision.py
│   │   ├── code_embedded.py
│   │   ├── documentation.py
│   │   └── ui_design.py
│   └── storage/         # Persistence
│       ├── provenance.py
│       └── work_product.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_imports.py -v
```

## Development

### Code Style

The project uses:
- Black for formatting
- Ruff for linting
- MyPy for type checking

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## License

MIT License