"""
Abstract base class for model clients.

Defines the interface that all model client implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResponse:
    """Response from a model generation request."""

    content: str
    tokens_input: int
    tokens_output: int
    model: str
    finish_reason: str
    raw_response: Optional[dict] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the request."""
        return self.tokens_input + self.tokens_output


class BaseModelClient(ABC):
    """
    Abstract base class for model clients.

    All model client implementations (Ollama, OpenAI, Anthropic) must
    inherit from this class and implement the abstract methods.
    """

    def __init__(self, endpoint: str, model_name: str):
        """
        Initialize the model client.

        Args:
            endpoint: Base URL for the model API
            model_name: Name of the model to use
        """
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional list of sequences to stop generation

        Returns:
            ModelResponse containing the generated content and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
    ):
        """
        Generate a streaming response from the model.

        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional list of sequences to stop generation

        Yields:
            Chunks of generated content as strings
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    def estimate_context_usage(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> int:
        """
        Estimate total context usage for a request.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Reserved tokens for output

        Returns:
            Estimated total context tokens (input + reserved output)
        """
        input_tokens = self.count_tokens(prompt)
        if system_prompt:
            input_tokens += self.count_tokens(system_prompt)
        return input_tokens + max_tokens

    async def health_check(self) -> bool:
        """
        Check if the model endpoint is healthy.

        Returns:
            True if endpoint is responsive, False otherwise
        """
        try:
            # Minimal generation to test connectivity
            response = await self.generate(
                prompt="Hi",
                max_tokens=1,
                temperature=0.0,
            )
            return response.content is not None
        except Exception:
            return False