"""
Model client abstractions for DATS.

Provides unified interfaces for different model providers:
- Ollama (local models)
- OpenAI-compatible (vLLM)
- Anthropic (frontier models)
"""

from src.models.base import BaseModelClient, ModelResponse
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient

__all__ = [
    "BaseModelClient",
    "ModelResponse",
    "OllamaClient",
    "OpenAICompatibleClient",
    "AnthropicClient",
]