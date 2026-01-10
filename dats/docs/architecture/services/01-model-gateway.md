# Model Gateway Service

> **DATS Microservice** - Unified LLM Interface  
> Priority: P0 (First to Extract)  
> Team: Platform  
> Status: Planned

---

## Overview

### Purpose

The Model Gateway Service provides a unified interface for all LLM interactions across DATS. It abstracts provider-specific implementations (Ollama, OpenAI, Anthropic, vLLM) behind a consistent API, enabling:

- **Provider Agnosticism**: Switch models without changing client code
- **Rate Limiting**: Centralized request throttling
- **Failover**: Automatic fallback to backup models
- **Observability**: Unified tracing for all LLM calls
- **Cost Tracking**: Centralized token usage metrics

### Current State (Monolith)

```
src/models/
├── base.py              # BaseModelClient abstract class
├── ollama_client.py     # Ollama API wrapper
├── openai_client.py     # OpenAI/vLLM wrapper
├── anthropic_client.py  # Anthropic wrapper
└── mock_client.py       # Testing mock
```

**Problems:**
- Each agent/worker imports clients directly
- No centralized rate limiting
- No automatic failover
- Duplicate tracing setup in each client

---

## API Specification

### Base URL

```
http://model-gateway:8000/api/v1
```

### Endpoints

#### POST /generate

Generate text completion from an LLM.

**Request:**
```json
{
  "model": "gemma3:12b",
  "prompt": "Write a Python function...",
  "system_prompt": "You are an expert developer.",
  "temperature": 0.7,
  "max_tokens": 2000,
  "stop_sequences": ["```"],
  "metadata": {
    "task_id": "abc123",
    "agent": "code_general",
    "tier": "small"
  }
}
```

**Response:**
```json
{
  "id": "gen-550e8400-e29b-41d4-a716-446655440000",
  "model": "gemma3:12b",
  "provider": "ollama",
  "content": "def fibonacci(n: int) -> int:\n    ...",
  "tokens_input": 150,
  "tokens_output": 200,
  "latency_ms": 1250,
  "finish_reason": "stop",
  "metadata": {
    "task_id": "abc123",
    "trace_id": "d81215d87cd927d7"
  }
}
```

**Error Response:**
```json
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "Model gemma3:12b is not available",
    "details": {
      "provider": "ollama",
      "attempted_fallbacks": ["gemma3:4b"],
      "trace_id": "d81215d87cd927d7"
    }
  }
}
```

#### GET /models

List available models and their status.

**Response:**
```json
{
  "models": [
    {
      "name": "gemma3:4b",
      "provider": "ollama",
      "tier": "tiny",
      "status": "available",
      "context_window": 8192,
      "capabilities": ["text-generation"]
    },
    {
      "name": "gemma3:12b",
      "provider": "ollama",
      "tier": "small",
      "status": "available",
      "context_window": 16384,
      "capabilities": ["text-generation"]
    },
    {
      "name": "claude-sonnet-4-20250514",
      "provider": "anthropic",
      "tier": "frontier",
      "status": "available",
      "context_window": 200000,
      "capabilities": ["text-generation", "vision"]
    }
  ]
}
```

#### GET /models/{name}

Get detailed info for a specific model.

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "providers": {
    "ollama": {
      "status": "connected",
      "endpoint": "http://192.168.1.11:11434",
      "models_available": 3
    },
    "anthropic": {
      "status": "connected",
      "rate_limit_remaining": 950
    }
  }
}
```

#### GET /metrics

Prometheus-compatible metrics.

---

## Data Models

### GenerateRequest

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GenerateRequest(BaseModel):
    """Request to generate text from an LLM."""
    
    model: str = Field(..., description="Model name or alias")
    prompt: str = Field(..., description="User prompt")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=100000)
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Pass-through metadata for tracing"
    )
    
    # Advanced options
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
```

### GenerateResponse

```python
class GenerateResponse(BaseModel):
    """Response from LLM generation."""
    
    id: str = Field(..., description="Unique generation ID")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider (ollama, anthropic, etc.)")
    content: str = Field(..., description="Generated text")
    tokens_input: int = Field(..., description="Input token count")
    tokens_output: int = Field(..., description="Output token count")
    latency_ms: int = Field(..., description="Generation latency")
    finish_reason: str = Field(..., description="Why generation stopped")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### ModelInfo

```python
class ModelInfo(BaseModel):
    """Information about an available model."""
    
    name: str
    provider: str
    tier: str  # tiny, small, large, frontier
    status: str  # available, unavailable, rate_limited
    context_window: int
    capabilities: List[str]
    rate_limit: Optional[dict] = None
```

---

## Architecture

### Internal Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL GATEWAY SERVICE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests from other services          │
│  │  Router     │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────▼──────┐                                                │
│  │   Request   │  - Validate request                            │
│  │  Validator  │  - Resolve model aliases                       │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────▼──────┐                                                │
│  │    Rate     │  - Per-model rate limiting                     │
│  │   Limiter   │  - Per-provider rate limiting                  │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────▼──────┐                                                │
│  │   Router    │  - Select provider for model                   │
│  │   Logic     │  - Handle failover                             │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                   PROVIDER CLIENTS                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │  Ollama  │  │  OpenAI  │  │Anthropic │  │   vLLM   │ │   │
│  │  │  Client  │  │  Client  │  │  Client  │  │  Client  │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    OBSERVABILITY                         │    │
│  │  - OpenTelemetry tracing                                 │    │
│  │  - Prometheus metrics                                    │    │
│  │  - Structured logging                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/model-gateway.yaml
providers:
  ollama:
    enabled: true
    endpoints:
      - host: http://192.168.1.11:11434
        priority: 1
        models: ["gemma3:4b", "gemma3:12b"]
      - host: http://192.168.1.12:11434
        priority: 2
        models: ["gemma3:12b"]
    timeout_seconds: 120
    
  anthropic:
    enabled: true
    api_key_env: ANTHROPIC_API_KEY
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 100000
    
  openai_compatible:
    enabled: true
    endpoint: http://192.168.1.11:8000/v1
    models: ["gpt-oss-20b"]

model_aliases:
  # Tier-based aliases
  tiny: gemma3:4b
  small: gemma3:12b
  large: gpt-oss-20b
  frontier: claude-sonnet-4-20250514
  
  # Capability-based aliases
  code: gemma3:12b
  reasoning: claude-sonnet-4-20250514

failover:
  enabled: true
  strategies:
    - type: same_tier
      description: Try another model in same tier
    - type: tier_up
      description: Escalate to higher tier
    - type: provider_fallback
      description: Try same model on different provider

rate_limiting:
  enabled: true
  default_rpm: 100
  default_tpm: 50000
  per_model: {}
```

---

## Migration Path

### Phase 1: Extract Clients (Week 1)

1. Create `services/model-gateway/` directory structure:
   ```
   services/model-gateway/
   ├── Dockerfile
   ├── requirements.txt
   ├── src/
   │   ├── __init__.py
   │   ├── main.py          # FastAPI app
   │   ├── config.py        # Settings
   │   ├── routers/
   │   │   ├── __init__.py
   │   │   ├── generate.py
   │   │   ├── models.py
   │   │   └── health.py
   │   ├── providers/
   │   │   ├── __init__.py
   │   │   ├── base.py
   │   │   ├── ollama.py
   │   │   ├── anthropic.py
   │   │   └── openai.py
   │   ├── services/
   │   │   ├── __init__.py
   │   │   ├── router.py    # Model routing logic
   │   │   ├── rate_limiter.py
   │   │   └── failover.py
   │   └── models/
   │       ├── __init__.py
   │       ├── requests.py
   │       └── responses.py
   └── tests/
   ```

2. Copy existing client code from `src/models/`:
   - `ollama_client.py` → `providers/ollama.py`
   - `anthropic_client.py` → `providers/anthropic.py`
   - `openai_client.py` → `providers/openai.py`

3. Adapt clients to use shared interface

### Phase 2: Create HTTP Client (Week 2)

Add to `dats-common`:

```python
# dats_common/clients/model_gateway.py
import httpx
from typing import Optional, Dict, Any

class ModelGatewayClient:
    """Client for Model Gateway Service."""
    
    def __init__(
        self,
        base_url: str = "http://model-gateway:8000/api/v1",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text from an LLM."""
        response = await self.client.post(
            f"{self.base_url}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = await self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    async def health(self) -> Dict[str, Any]:
        """Check service health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

### Phase 3: Update Consumers (Week 3)

Update `src/agents/base.py` to use the client:

```python
# Before (direct import)
from src.models.ollama_client import OllamaClient
client = OllamaClient(...)
response = await client.generate(prompt)

# After (gateway client)
from dats_common.clients.model_gateway import ModelGatewayClient
client = ModelGatewayClient()
response = await client.generate(model="small", prompt=prompt)
```

### Phase 4: Deploy & Switch (Week 4)

1. Deploy Model Gateway as Docker container
2. Configure environment variable: `MODEL_GATEWAY_URL`
3. Feature flag: `USE_MODEL_GATEWAY=true`
4. Gradual rollout: 10% → 50% → 100%
5. Remove direct client imports from monolith

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dats-common
COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Observability

### Traces

All requests are traced with OpenTelemetry:

```python
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    tracer = get_tracer("model-gateway")
    with tracer.start_as_current_span(
        f"generate.{request.state.model}",
        attributes={
            "model.name": request.state.model,
            "model.provider": request.state.provider,
            "request.tokens_estimate": len(request.state.prompt) // 4,
        },
    ) as span:
        response = await call_next(request)
        span.set_attribute("response.tokens_output", response.tokens_output)
        return response
```

### Metrics (Prometheus)

```
# Request counts
model_gateway_requests_total{model="gemma3:12b", provider="ollama", status="success"}
model_gateway_requests_total{model="gemma3:12b", provider="ollama", status="error"}

# Latency histogram
model_gateway_latency_seconds{model="gemma3:12b", provider="ollama", quantile="0.5"}
model_gateway_latency_seconds{model="gemma3:12b", provider="ollama", quantile="0.95"}

# Token counts
model_gateway_tokens_total{model="gemma3:12b", direction="input"}
model_gateway_tokens_total{model="gemma3:12b", direction="output"}

# Rate limiting
model_gateway_rate_limit_remaining{model="gemma3:12b"}
model_gateway_rate_limit_rejections_total{model="gemma3:12b"}
```

### Logs

Structured JSON logging:

```json
{
  "timestamp": "2026-01-09T18:00:00Z",
  "level": "INFO",
  "service": "model-gateway",
  "trace_id": "d81215d87cd927d7",
  "span_id": "a5420fad319a288f",
  "event": "generation_complete",
  "model": "gemma3:12b",
  "provider": "ollama",
  "tokens_input": 150,
  "tokens_output": 200,
  "latency_ms": 1250,
  "task_id": "abc123"
}
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_router.py
import pytest
from src.services.router import ModelRouter

@pytest.fixture
def router():
    return ModelRouter(config_path="tests/fixtures/config.yaml")

def test_resolve_alias(router):
    assert router.resolve_model("small") == "gemma3:12b"
    
def test_select_provider(router):
    provider = router.select_provider("gemma3:12b")
    assert provider.name == "ollama"

def test_failover_same_tier(router):
    # Simulate first provider failure
    router.mark_unavailable("gemma3:12b", "ollama-1")
    provider = router.select_provider("gemma3:12b")
    assert provider.endpoint == "ollama-2"
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from httpx import AsyncClient
from src.main import app

@pytest.mark.asyncio
async def test_generate_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/generate",
            json={
                "model": "gemma3:4b",
                "prompt": "Hello, world!",
                "max_tokens": 50,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["model"] == "gemma3:4b"
```

### Load Tests

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class ModelGatewayUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate_small(self):
        self.client.post(
            "/api/v1/generate",
            json={
                "model": "small",
                "prompt": "Write a haiku about coding.",
                "max_tokens": 100,
            },
        )
    
    @task
    def list_models(self):
        self.client.get("/api/v1/models")
```

---

## Dependencies

### From dats-common

- `dats_common.telemetry` - OpenTelemetry setup
- `dats_common.config` - Base settings

### External

```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Providers
anthropic>=0.7.0
openai>=1.3.0

# Observability
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
opentelemetry-exporter-otlp>=1.21.0
prometheus-client>=0.19.0

# Rate limiting
limits>=3.6.0
redis>=5.0.0  # For distributed rate limiting
```

---

## Environment Variables

```bash
# Provider endpoints
OLLAMA_HOST=http://192.168.1.11:11434
OLLAMA_HOST_SECONDARY=http://192.168.1.12:11434
VLLM_ENDPOINT=http://192.168.1.11:8000/v1
ANTHROPIC_API_KEY=sk-ant-...

# Service config
MODEL_GATEWAY_PORT=8000
MODEL_GATEWAY_CONFIG=/app/config/model-gateway.yaml
LOG_LEVEL=INFO

# Telemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_SERVICE_NAME=model-gateway

# Rate limiting
REDIS_URL=redis://redis:6379/0
RATE_LIMIT_ENABLED=true
```

---

## Success Criteria

Before marking complete:

- [ ] All existing tests pass with gateway
- [ ] P95 latency within 10% of direct client calls
- [ ] Failover works for provider outages
- [ ] Rate limiting prevents quota exhaustion
- [ ] Metrics visible in Grafana
- [ ] Traces visible in Jaeger
- [ ] Zero direct model client imports in agents/workers

---

*Last updated: January 2026*