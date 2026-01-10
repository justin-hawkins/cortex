# File: services/model-gateway/src/providers/anthropic.py
"""Anthropic provider implementation."""

import time
from typing import Any, AsyncIterator

import anthropic

from src.providers.base import (
    BaseProvider,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    StreamChunk,
)


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic API (Claude models).
    
    Uses the official Anthropic Python SDK for API access.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        api_key: str = "",
        timeout: float = 300.0,
        configured_models: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            name: Provider instance name
            endpoint: Anthropic API endpoint (usually https://api.anthropic.com/v1)
            api_key: Anthropic API key
            timeout: Request timeout in seconds
            configured_models: List of model configs from model-gateway.yaml
        """
        super().__init__(name, endpoint, timeout)
        self._api_key = api_key
        self._client: anthropic.AsyncAnthropic | None = None
        self._configured_models = configured_models or []

    @property
    def provider_type(self) -> str:
        return "anthropic"

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Generate a response using Anthropic API."""
        if not self._api_key:
            raise ValueError("Anthropic API key is not configured")

        start_time = time.time()
        client = self._get_client()

        # Build messages list
        messages = [{"role": "user", "content": request.prompt}]

        # Make API call
        response = await client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=messages,
            system=request.system_prompt if request.system_prompt else anthropic.NOT_GIVEN,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences if request.stop_sequences else anthropic.NOT_GIVEN,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract content from response
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        return GenerateResponse(
            model=response.model,
            provider=self.name,
            content=content,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason or "stop",
            metadata={
                "id": response.id,
                "type": response.type,
            },
        )

    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using Anthropic API."""
        if not self._api_key:
            raise ValueError("Anthropic API key is not configured")

        from uuid import uuid4
        
        client = self._get_client()
        chunk_id = f"gen-{uuid4()}"

        # Build messages list
        messages = [{"role": "user", "content": request.prompt}]

        # Stream API call
        async with client.messages.stream(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=messages,
            system=request.system_prompt if request.system_prompt else anthropic.NOT_GIVEN,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences if request.stop_sequences else anthropic.NOT_GIVEN,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(
                    id=chunk_id,
                    content=text,
                    done=False,
                )
            
            # Final chunk with completion info
            final_message = await stream.get_final_message()
            yield StreamChunk(
                id=chunk_id,
                content="",
                done=True,
                finish_reason=final_message.stop_reason or "stop",
                tokens_output=final_message.usage.output_tokens,
            )

    async def list_models(self) -> list[ModelInfo]:
        """List models available from Anthropic."""
        # Anthropic doesn't have a models list endpoint,
        # so we return the configured models
        return [
            ModelInfo(
                name=m["name"],
                provider=self.provider_type,
                endpoint_name=self.name,
                context_window=m.get("context_window", 200000),
                tier=m.get("tier", "frontier"),
                model_type=m.get("type", "chat"),
                description=m.get("description", ""),
                status="available" if self._api_key else "no_api_key",
                metadata={},
            )
            for m in self._configured_models
        ]

    async def health_check(self) -> bool:
        """Check if the Anthropic API is accessible."""
        if not self._api_key:
            self._healthy = False
            return False

        try:
            # Make a minimal request to check API access
            client = self._get_client()
            response = await client.messages.create(
                model="claude-haiku-3-20240307",  # Use cheapest model
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            self._healthy = response is not None
            return self._healthy
        except anthropic.AuthenticationError:
            self._healthy = False
            return False
        except Exception:
            self._healthy = False
            return False