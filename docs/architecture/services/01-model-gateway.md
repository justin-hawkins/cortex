# File: docs/architecture/services/01-model-gateway.md
# Model Gateway Service

> **Priority**: P0 (First to Extract) | **Team**: Platform | **Status**: Planned
>
> Unified LLM interface abstracting Ollama, OpenAI, Anthropic, and vLLM.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/model-gateway/contracts/openapi.yaml` |
| AsyncAPI | N/A (stateless service) |

---

## Purpose

Abstracts LLM providers behind a consistent API:
- **Provider Agnosticism**: Switch models without changing client code
- **Rate Limiting**: Centralized request throttling (Redis-backed)
- **Failover**: Automatic fallback to backup models
- **Cost Tracking**: Centralized token usage metrics

---

## API Endpoints

### POST /generate

Generate text completion from an LLM.

```json
// Request
{
  "model": "gemma3:12b",
  "prompt": "Write a Python function...",
  "system_prompt": "You are an expert developer.",
  "temperature": 0.7,
  "max_tokens": 2000,
  "metadata": {"task_id": "abc123"}
}

// Response
{
  "id": "gen-550e8400-...",
  "model": "gemma3:12b",
  "provider": "ollama",
  "content": "def fibonacci(n: int) -> int:\n    ...",
  "tokens_input": 150,
  "tokens_output": 200,
  "latency_ms": 1250,
  "finish_reason": "stop"
}
```

### GET /models

List available models with status, tier, and capabilities.

### GET /models/{name}

Get detailed info for a specific model.

---

## Configuration

```yaml
# config/model-gateway.yaml
providers:
  ollama:
    endpoints:
      - name: ollama_cpu_large
        host: http://192.168.1.11:11434
        mode: cpu
        models: [qwen3-coder:30b-a3b-q8_0-64k]
      - name: ollama_gpu_general
        host: http://192.168.1.12:11434
        mode: gpu
        models: [gemma3:4b, gemma3:12b, mxbai-embed-large:335m]
  anthropic:
    endpoint: https://api.anthropic.com/v1
    rate_limit: {requests_per_minute: 60}
  vllm:
    endpoint: http://192.168.1.11:8000/v1
    models: [openai/gpt-oss-20b]

model_aliases:
  tiny: gemma3:4b
  small: gemma3:12b
  large: openai/gpt-oss-20b
  frontier: claude-sonnet-4-20250514
  coding: qwen3-coder:30b-a3b-q8_0-64k
  embedding: mxbai-embed-large:335m

failover:
  strategies: [same_tier, tier_up, provider_fallback]
```

---

## Data Models

```python
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=100000)
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    id: str
    model: str
    provider: str
    content: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    finish_reason: str
    metadata: Dict[str, Any] = {}
```

---

## Client (dats-common)

```python
from dats_common.clients.model_gateway import ModelGatewayClient

client = ModelGatewayClient()
response = await client.generate(model="small", prompt="Hello")
models = await client.list_models()
```

---

## Migration Path

1. **Week 1**: Create service, copy clients from `src/models/`
2. **Week 2**: Add `ModelGatewayClient` to dats-common
3. **Week 3**: Update agents/workers to use client
4. **Week 4**: Deploy, gradual traffic switch (10% → 50% → 100%)

---

## Environment Variables

```bash
OLLAMA_CPU_LARGE=http://192.168.1.11:11434
OLLAMA_GPU_GENERAL=http://192.168.1.12:11434
VLLM_ENDPOINT=http://192.168.1.11:8000/v1
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://192.168.1.44:6379/0
RATE_LIMIT_ENABLED=true
```

---

## Success Criteria

- [ ] All existing tests pass with gateway
- [ ] P95 latency within 10% of direct client calls
- [ ] Failover works for provider outages
- [ ] Rate limiting prevents quota exhaustion
- [ ] Zero direct model client imports in agents/workers
