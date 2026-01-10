# File: services/model-gateway/src/providers/vllm.py
"""vLLM provider implementation (OpenAI-compatible API)."""

import json
import time
from typing import Any, AsyncIterator

import httpx

from src.providers.base import (
    BaseProvider,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    StreamChunk,
)


class VLLMProvider(BaseProvider):
    """
    Provider for vLLM with OpenAI-compatible API.
    
    vLLM exposes an OpenAI-compatible endpoint at /v1/completions
    and /v1/chat/completions.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout: float = 300.0,
        configured_models: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize vLLM provider.
        
        Args:
            name: Provider instance name
            endpoint: vLLM API endpoint (e.g., http://192.168.1.11:8000/v1)
            timeout: Request timeout in seconds
            configured_models: List of model configs from model-gateway.yaml
        """
        super().__init__(name, endpoint, timeout)
        self._client: httpx.AsyncClient | None = None
        self._configured_models = configured_models or []

    @property
    def provider_type(self) -> str:
        return "vllm"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Generate a response using vLLM OpenAI-compatible API."""
        start_time = time.time()
        client = await self._get_client()

        # Use chat completions API for system prompt support
        if request.system_prompt:
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ]
            payload: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            response = await client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
            )
        else:
            # Use completions API for simple prompts
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            response = await client.post(
                f"{self.endpoint}/completions",
                json=payload,
            )

        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract content based on API type
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
            else:
                content = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
        else:
            content = ""
            finish_reason = "error"

        # Extract usage info
        usage = data.get("usage", {})

        return GenerateResponse(
            model=data.get("model", request.model),
            provider=self.name,
            content=content,
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            metadata={
                "endpoint": self.endpoint,
                "id": data.get("id"),
            },
        )

    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using vLLM OpenAI-compatible API."""
        from uuid import uuid4
        
        client = await self._get_client()
        chunk_id = f"gen-{uuid4()}"

        # Use chat completions API for system prompt support
        if request.system_prompt:
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ]
            payload: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            url = f"{self.endpoint}/chat/completions"
        else:
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            url = f"{self.endpoint}/completions"

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(
                            id=chunk_id,
                            content="",
                            done=True,
                            finish_reason="stop",
                        )
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            
                            # Extract content (differs between chat and completions)
                            if "delta" in choice:
                                content = choice["delta"].get("content", "")
                            else:
                                content = choice.get("text", "")
                            
                            finish_reason = choice.get("finish_reason")
                            done = finish_reason is not None
                            
                            yield StreamChunk(
                                id=chunk_id,
                                content=content,
                                done=done,
                                finish_reason=finish_reason,
                            )
                    except json.JSONDecodeError:
                        continue

    async def list_models(self) -> list[ModelInfo]:
        """List models available from this vLLM endpoint."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/models")
            response.raise_for_status()
            data = response.json()
            
            # Create a lookup from configured models
            config_lookup = {
                m["name"]: m for m in self._configured_models
            }
            
            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                
                # Get config if available
                config = config_lookup.get(model_id, {})
                
                models.append(ModelInfo(
                    name=model_id,
                    provider=self.provider_type,
                    endpoint_name=self.name,
                    context_window=config.get("context_window", 32768),
                    tier=config.get("tier", "large"),
                    model_type=config.get("type", "chat"),
                    description=config.get("description", ""),
                    status="available",
                    metadata={
                        "owned_by": model_data.get("owned_by"),
                        "created": model_data.get("created"),
                    },
                ))
            
            return models
        except Exception as e:
            # Return configured models with unknown status on error
            return [
                ModelInfo(
                    name=m["name"],
                    provider=self.provider_type,
                    endpoint_name=self.name,
                    context_window=m.get("context_window", 32768),
                    tier=m.get("tier", "large"),
                    model_type=m.get("type", "chat"),
                    description=m.get("description", ""),
                    status="unknown",
                    metadata={"error": str(e)},
                )
                for m in self._configured_models
            ]

    async def health_check(self) -> bool:
        """Check if the vLLM endpoint is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.endpoint}/models",
                timeout=10.0,
            )
            self._healthy = response.status_code == 200
            return self._healthy
        except Exception:
            self._healthy = False
            return False