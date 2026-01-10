# File: services/model-gateway/src/providers/ollama.py
"""Ollama provider implementation."""

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


class OllamaProvider(BaseProvider):
    """
    Provider for Ollama API.
    
    Supports text generation and model listing from Ollama endpoints.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout: float = 300.0,
        configured_models: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize Ollama provider.
        
        Args:
            name: Provider instance name
            endpoint: Ollama API endpoint (e.g., http://192.168.1.12:11434)
            timeout: Request timeout in seconds
            configured_models: List of model configs from model-gateway.yaml
        """
        super().__init__(name, endpoint, timeout)
        self._client: httpx.AsyncClient | None = None
        self._configured_models = configured_models or []

    @property
    def provider_type(self) -> str:
        return "ollama"

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
        """Generate a response using Ollama API."""
        start_time = time.time()
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences

        response = await client.post(
            f"{self.endpoint}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = int((time.time() - start_time) * 1000)

        return GenerateResponse(
            model=data.get("model", request.model),
            provider=self.name,
            content=data.get("response", ""),
            tokens_input=data.get("prompt_eval_count", 0),
            tokens_output=data.get("eval_count", 0),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", "stop"),
            metadata={
                "endpoint": self.endpoint,
                "raw_response": data,
            },
        )

    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using Ollama API."""
        from uuid import uuid4
        
        client = await self._get_client()
        chunk_id = f"gen-{uuid4()}"

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences

        async with client.stream(
            "POST",
            f"{self.endpoint}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    done = data.get("done", False)
                    yield StreamChunk(
                        id=chunk_id,
                        content=data.get("response", ""),
                        done=done,
                        finish_reason=data.get("done_reason") if done else None,
                        tokens_output=data.get("eval_count") if done else None,
                    )

    async def list_models(self) -> list[ModelInfo]:
        """List models available from this Ollama endpoint."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Create a lookup from configured models
            config_lookup = {
                m["name"]: m for m in self._configured_models
            }
            
            models = []
            for model_data in data.get("models", []):
                model_name = model_data.get("name", "")
                
                # Get config if available
                config = config_lookup.get(model_name, {})
                
                models.append(ModelInfo(
                    name=model_name,
                    provider=self.provider_type,
                    endpoint_name=self.name,
                    context_window=config.get("context_window", 32768),
                    tier=config.get("tier", "small"),
                    model_type=config.get("type", "chat"),
                    description=config.get("description", ""),
                    status="available",
                    metadata={
                        "size": model_data.get("size"),
                        "modified_at": model_data.get("modified_at"),
                        "digest": model_data.get("digest"),
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
                    tier=m.get("tier", "small"),
                    model_type=m.get("type", "chat"),
                    description=m.get("description", ""),
                    status="unknown",
                    metadata={"error": str(e)},
                )
                for m in self._configured_models
            ]

    async def health_check(self) -> bool:
        """Check if the Ollama endpoint is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.endpoint}/api/tags",
                timeout=10.0,
            )
            self._healthy = response.status_code == 200
            return self._healthy
        except Exception:
            self._healthy = False
            return False

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model: Optional model override (defaults to configured embedding model)
            
        Returns:
            List of embedding floats
        """
        client = await self._get_client()
        
        # Find an embedding model from configured models
        if model is None:
            for m in self._configured_models:
                if m.get("type") == "embedding":
                    model = m["name"]
                    break
        
        if model is None:
            model = "mxbai-embed-large:335m"  # Default fallback

        response = await client.post(
            f"{self.endpoint}/api/embeddings",
            json={
                "model": model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()

        return data.get("embedding", [])