# File: services/model-gateway/README.md
# Model Gateway Service

> **DATS Microservices** - Unified LLM Interface
> 
> Abstracts Ollama, vLLM, and Anthropic behind a consistent API.

## Overview

The Model Gateway provides:
- **Provider Agnosticism**: Switch models without changing client code
- **Unified API**: Single endpoint for all LLM providers
- **Streaming Support**: Server-Sent Events for real-time responses
- **Model Discovery**: List available models with status and capabilities
- **Failover**: Automatic fallback to backup models (future)
- **Cost Tracking**: Centralized token usage metrics

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Run with hot reload
make run
```

### Docker

```bash
# Build and run
make run-docker

# With observability (Jaeger)
make run-full
```

### Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Edit with your values (especially ANTHROPIC_API_KEY)
vim .env
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate` | Generate text completion |
| POST | `/api/v1/generate` (stream=true) | Streaming completion via SSE |
| GET | `/api/v1/models` | List available models |
| GET | `/api/v1/models/{name}` | Get model details |
| GET | `/api/v1/health` | Basic health check |
| GET | `/api/v1/health/detailed` | Detailed health with providers |

## Usage Examples

### Generate Text

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:12b",
    "prompt": "Write a Python function to calculate fibonacci",
    "temperature": 0.7,
    "max_tokens": 2000
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:12b",
    "prompt": "Explain quantum computing",
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8000/api/v1/models
```

### Model Aliases

For convenience, you can use aliases instead of full model names:

| Alias | Model | Use Case |
|-------|-------|----------|
| `tiny` | gemma3:4b | Fast inference |
| `small` | gemma3:12b | Balanced |
| `large` | openai/gpt-oss-20b | Higher quality |
| `frontier` | claude-sonnet-4-20250514 | Best quality |
| `coding` | qwen3-coder:30b-a3b-q8_0-64k | Code generation |
| `embedding` | mxbai-embed-large:335m | Text embeddings |

## Configuration

### Provider Endpoints

Configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_CPU_ENDPOINT` | `http://192.168.1.11:11434` | Ollama CPU server |
| `OLLAMA_GPU_ENDPOINT` | `http://192.168.1.12:11434` | Ollama GPU server |
| `VLLM_ENDPOINT` | `http://192.168.1.11:8000/v1` | vLLM server |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_PORT` | `8000` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `OTEL_SERVICE_NAME` | `model-gateway` | OpenTelemetry service name |

## Development

### Project Structure

```
services/model-gateway/
├── src/
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings & configuration
│   ├── routers/             # API endpoints
│   │   ├── health.py
│   │   ├── generate.py
│   │   └── models.py
│   ├── services/            # Business logic
│   │   ├── model_registry.py
│   │   └── failover.py
│   └── providers/           # LLM provider clients
│       ├── base.py
│       ├── ollama.py
│       ├── vllm.py
│       └── anthropic.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── config/
│   └── model-gateway.yaml
└── contracts/
    └── openapi.yaml
```

### Commands

```bash
make test           # Run all tests
make test-unit      # Run unit tests only
make lint           # Lint code
make format         # Format code
make contract-test  # Run contract tests
make clean          # Clean up cache files
```

## Infrastructure

This service connects to:

| Resource | Endpoint | Purpose |
|----------|----------|---------|
| Ollama (CPU) | 192.168.1.11:11434 | Large coding models |
| Ollama (GPU) | 192.168.1.12:11434 | General inference |
| vLLM | 192.168.1.11:8000/v1 | High-throughput inference |
| Anthropic | api.anthropic.com | Frontier models |

## Related Documentation

- [Service Specification](../../docs/architecture/services/01-model-gateway.md)
- [Server Configuration](../../docs/architecture/servers.yaml)
- [Service Common Patterns](../../docs/architecture/_shared/SERVICE_COMMON.md)