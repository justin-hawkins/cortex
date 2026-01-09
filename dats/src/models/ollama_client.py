"""
Ollama model client implementation.

Provides async interface to Ollama API for local model inference.
"""

import json
from typing import AsyncIterator, Optional

import httpx

from src.models.base import BaseModelClient, ModelResponse
from src.telemetry.llm_tracer import LLMCallTracer


class OllamaClient(BaseModelClient):
    """
    Client for Ollama API.

    Supports both generation and embedding endpoints.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        timeout: float = 300.0,
    ):
        """
        Initialize Ollama client.

        Args:
            endpoint: Ollama API endpoint (e.g., http://192.168.1.79:11434)
            model_name: Model name (e.g., gemma3:4b)
            timeout: Request timeout in seconds
        """
        super().__init__(endpoint, model_name)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._llm_tracer = LLMCallTracer(provider="ollama", model=model_name)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        prompt_template: Optional[str] = None,
        context_items: int = 0,
    ) -> ModelResponse:
        """
        Generate a response using Ollama API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            prompt_template: Optional prompt template name/version for tracing
            context_items: Number of RAG context items included (for tracing)

        Returns:
            ModelResponse with generated content
        """
        # Use LLM tracer to capture full prompt/response telemetry
        with self._llm_tracer.trace_call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            prompt_template=prompt_template,
            context_items=context_items,
        ) as call_recorder:
            client = await self._get_client()

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            if stop_sequences:
                payload["options"]["stop"] = stop_sequences

            response = await client.post(
                f"{self.endpoint}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            model_response = ModelResponse(
                content=data.get("response", ""),
                tokens_input=data.get("prompt_eval_count", 0),
                tokens_output=data.get("eval_count", 0),
                model=data.get("model", self.model_name),
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )

            # Record the response in telemetry
            call_recorder.record_from_model_response(model_response)

            return model_response

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Ollama API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences

        Yields:
            Chunks of generated content
        """
        client = await self._get_client()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        async with client.stream(
            "POST",
            f"{self.endpoint}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Ollama's tokenize endpoint.

        Note: This is a synchronous approximation. For accurate counts,
        use the async version or rely on response token counts.

        Args:
            text: Text to tokenize

        Returns:
            Approximate token count (4 chars per token estimate)
        """
        # Ollama doesn't have a public tokenize API, so estimate
        # Most models average around 4 characters per token
        return len(text) // 4

    async def count_tokens_async(self, text: str) -> int:
        """
        Count tokens using Ollama API (if available).

        Some Ollama versions support /api/tokenize endpoint.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.endpoint}/api/tokenize",
                json={"model": self.model_name, "prompt": text},
            )
            if response.status_code == 200:
                data = response.json()
                return len(data.get("tokens", []))
        except Exception:
            pass

        # Fall back to estimate
        return self.count_tokens(text)

    async def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            List of embedding floats
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.endpoint}/api/embeddings",
            json={
                "model": self.model_name,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()

        return data.get("embedding", [])

    async def list_models(self) -> list[dict]:
        """
        List available models on this Ollama instance.

        Returns:
            List of model info dictionaries
        """
        client = await self._get_client()
        response = await client.get(f"{self.endpoint}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])