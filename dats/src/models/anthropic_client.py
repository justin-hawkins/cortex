"""
Anthropic model client implementation.

Provides async interface to Anthropic API for frontier model inference.
"""

from typing import AsyncIterator, Optional

import anthropic
import tiktoken

from src.models.base import BaseModelClient, ModelResponse
from src.telemetry.llm_tracer import LLMCallTracer


class AnthropicClient(BaseModelClient):
    """
    Client for Anthropic API.

    Uses the official anthropic SDK for Claude models.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
    ):
        """
        Initialize Anthropic client.

        Args:
            endpoint: Anthropic API endpoint (typically https://api.anthropic.com/v1)
            model_name: Model name (e.g., claude-sonnet-4-20250514)
            api_key: Anthropic API key (required)
            timeout: Request timeout in seconds
        """
        super().__init__(endpoint, model_name)
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._llm_tracer = LLMCallTracer(provider="anthropic", model=model_name)

        # Use tiktoken for token estimation (Claude uses similar tokenization)
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
                )
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.close()
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
        Generate a response using Anthropic API.

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
            client = self._get_client()

            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = await client.messages.create(**kwargs)

            # Extract content from response
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            model_response = ModelResponse(
                content=content,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                model=response.model,
                finish_reason=response.stop_reason or "stop",
                raw_response=response.model_dump(),
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
        Generate a streaming response using Anthropic API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences

        Yields:
            Chunks of generated content
        """
        client = self._get_client()

        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (approximation for Claude).

        Note: This is an approximation. Anthropic's exact tokenization
        may differ slightly.

        Args:
            text: Text to tokenize

        Returns:
            Approximate token count
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback estimate
        return len(text) // 4

    async def count_tokens_exact(self, text: str) -> int:
        """
        Count tokens using Anthropic's token counting API.

        Args:
            text: Text to tokenize

        Returns:
            Exact token count from Anthropic
        """
        client = self._get_client()
        
        try:
            result = await client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            # Fall back to estimate
            return self.count_tokens(text)