"""
OpenAI-compatible model client implementation.

Provides async interface to OpenAI-compatible APIs (vLLM, etc.).
"""

import json
from typing import AsyncIterator, Optional

import httpx
import tiktoken

from src.models.base import BaseModelClient, ModelResponse


class OpenAICompatibleClient(BaseModelClient):
    """
    Client for OpenAI-compatible APIs.

    Works with vLLM and other OpenAI-compatible inference servers.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            endpoint: API endpoint (e.g., http://192.168.1.11:8000/v1)
            model_name: Model name (e.g., gpt-oss:20b)
            api_key: Optional API key (some servers don't require it)
            timeout: Request timeout in seconds
        """
        super().__init__(endpoint, model_name)
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # Try to get a tokenizer for token counting
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
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
    ) -> ModelResponse:
        """
        Generate a response using OpenAI-compatible API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences

        Returns:
            ModelResponse with generated content
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        response = await client.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ModelResponse(
            content=choice["message"]["content"],
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            model=data.get("model", self.model_name),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using OpenAI-compatible API.

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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        async with client.stream(
            "POST",
            f"{self.endpoint}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback estimate
        return len(text) // 4

    async def list_models(self) -> list[dict]:
        """
        List available models on this endpoint.

        Returns:
            List of model info dictionaries
        """
        client = await self._get_client()
        response = await client.get(f"{self.endpoint}/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])