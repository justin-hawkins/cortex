# File: services/model-gateway/src/providers/base.py
"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Type of model capability."""
    
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class ModelTier(str, Enum):
    """Model quality/size tier."""
    
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    FRONTIER = "frontier"
    CODING = "coding"


class FinishReason(str, Enum):
    """Reason for generation completion."""
    
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    model: str = Field(..., description="Model name or alias")
    prompt: str = Field(..., description="The prompt to generate from")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2000, ge=1, le=100000, description="Maximum tokens to generate")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: dict[str, Any] | None = Field(None, description="Custom metadata")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    id: str = Field(default_factory=lambda: f"gen-{uuid4()}")
    model: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="Provider that served the request")
    content: str = Field(..., description="Generated content")
    tokens_input: int = Field(0, description="Input token count")
    tokens_output: int = Field(0, description="Output token count")
    latency_ms: int = Field(0, description="Generation latency in milliseconds")
    finish_reason: str = Field("stop", description="Reason for completion")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.tokens_input + self.tokens_output


class StreamChunk(BaseModel):
    """A chunk of streamed response."""
    
    id: str
    content: str
    done: bool = False
    finish_reason: str | None = None
    tokens_output: int | None = None


@dataclass
class ModelInfo:
    """Information about an available model."""
    
    name: str
    provider: str
    endpoint_name: str
    context_window: int = 32768
    tier: str = "small"
    model_type: str = "chat"
    description: str = ""
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations (Ollama, vLLM, Anthropic) must inherit
    from this class and implement the abstract methods.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout: float = 300.0,
    ):
        """
        Initialize the provider.
        
        Args:
            name: Provider instance name (e.g., "ollama_cpu_large")
            endpoint: Base URL for the provider API
            timeout: Request timeout in seconds
        """
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._healthy = False

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Return the provider type (e.g., 'ollama', 'vllm', 'anthropic')."""
        pass

    @abstractmethod
    async def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """
        Generate a response from the model.
        
        Args:
            request: The generation request
            
        Returns:
            GenerateResponse containing the generated content and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a streaming response from the model.
        
        Args:
            request: The generation request
            
        Yields:
            StreamChunk objects containing partial content
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """
        List models available from this provider endpoint.
        
        Returns:
            List of ModelInfo objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider endpoint is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass

    async def close(self) -> None:
        """Clean up provider resources."""
        pass

    @property
    def is_healthy(self) -> bool:
        """Return cached health status."""
        return self._healthy