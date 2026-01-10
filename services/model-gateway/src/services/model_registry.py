# File: services/model-gateway/src/services/model_registry.py
"""Model registry service for managing providers and models."""

import asyncio
import logging
from typing import Any

from src.config import get_gateway_config, get_settings
from src.providers.base import BaseProvider, GenerateRequest, GenerateResponse, ModelInfo
from src.providers.ollama import OllamaProvider
from src.providers.vllm import VLLMProvider
from src.providers.anthropic import AnthropicProvider

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing LLM providers and their models.
    
    Responsibilities:
    - Initialize providers from configuration
    - Resolve model names (including aliases) to providers
    - Track model availability and health
    - Provide unified access to generation
    """

    def __init__(self) -> None:
        """Initialize the model registry."""
        self._providers: dict[str, BaseProvider] = {}
        self._model_to_provider: dict[str, BaseProvider] = {}
        self._aliases: dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize providers from configuration."""
        if self._initialized:
            return

        config = get_gateway_config()
        settings = get_settings()

        # Load model aliases
        self._aliases = config.get("model_aliases", {})

        # Initialize Ollama providers
        for endpoint_config in config.get("providers", {}).get("ollama", {}).get("endpoints", []):
            name = endpoint_config["name"]
            host = endpoint_config["host"]
            models = endpoint_config.get("models", [])
            
            provider = OllamaProvider(
                name=name,
                endpoint=host,
                timeout=settings.default_timeout,
                configured_models=models,
            )
            self._providers[name] = provider
            
            # Map models to this provider
            for model_config in models:
                model_name = model_config["name"]
                self._model_to_provider[model_name] = provider
            
            logger.info(f"Initialized Ollama provider: {name} at {host}")

        # Initialize vLLM providers
        for endpoint_config in config.get("providers", {}).get("vllm", {}).get("endpoints", []):
            name = endpoint_config["name"]
            host = endpoint_config["host"]
            models = endpoint_config.get("models", [])
            
            provider = VLLMProvider(
                name=name,
                endpoint=host,
                timeout=settings.default_timeout,
                configured_models=models,
            )
            self._providers[name] = provider
            
            # Map models to this provider
            for model_config in models:
                model_name = model_config["name"]
                self._model_to_provider[model_name] = provider
            
            logger.info(f"Initialized vLLM provider: {name} at {host}")

        # Initialize Anthropic providers
        for endpoint_config in config.get("providers", {}).get("anthropic", {}).get("endpoints", []):
            name = endpoint_config["name"]
            host = endpoint_config["host"]
            api_key = endpoint_config.get("api_key", settings.anthropic_api_key)
            models = endpoint_config.get("models", [])
            
            provider = AnthropicProvider(
                name=name,
                endpoint=host,
                api_key=api_key,
                timeout=settings.default_timeout,
                configured_models=models,
            )
            self._providers[name] = provider
            
            # Map models to this provider
            for model_config in models:
                model_name = model_config["name"]
                self._model_to_provider[model_name] = provider
            
            logger.info(f"Initialized Anthropic provider: {name}")

        self._initialized = True
        logger.info(f"Model registry initialized with {len(self._providers)} providers")

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
        self._model_to_provider.clear()
        self._initialized = False

    def resolve_model(self, model: str) -> str:
        """
        Resolve a model name or alias to the actual model name.
        
        Args:
            model: Model name or alias
            
        Returns:
            Resolved model name
        """
        return self._aliases.get(model, model)

    def get_provider_for_model(self, model: str) -> BaseProvider | None:
        """
        Get the provider instance for a given model.
        
        Args:
            model: Model name (resolved, not alias)
            
        Returns:
            Provider instance or None if not found
        """
        return self._model_to_provider.get(model)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate text using the appropriate provider.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response
            
        Raises:
            ValueError: If model is not found
        """
        # Resolve alias to actual model name
        resolved_model = self.resolve_model(request.model)
        
        # Get provider for this model
        provider = self.get_provider_for_model(resolved_model)
        if provider is None:
            raise ValueError(f"Model not found: {request.model} (resolved: {resolved_model})")

        # Update request with resolved model name
        resolved_request = GenerateRequest(
            model=resolved_model,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            stream=request.stream,
            metadata=request.metadata,
        )

        return await provider.generate(resolved_request)

    async def generate_stream(self, request: GenerateRequest):
        """
        Generate streaming text using the appropriate provider.
        
        Args:
            request: Generation request
            
        Yields:
            StreamChunk objects
            
        Raises:
            ValueError: If model is not found
        """
        # Resolve alias to actual model name
        resolved_model = self.resolve_model(request.model)
        
        # Get provider for this model
        provider = self.get_provider_for_model(resolved_model)
        if provider is None:
            raise ValueError(f"Model not found: {request.model} (resolved: {resolved_model})")

        # Update request with resolved model name
        resolved_request = GenerateRequest(
            model=resolved_model,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            stream=True,
            metadata=request.metadata,
        )

        async for chunk in provider.generate_stream(resolved_request):
            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        """
        List all available models from all providers.
        
        Returns:
            List of ModelInfo objects
        """
        all_models: list[ModelInfo] = []
        
        for provider in self._providers.values():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception as e:
                logger.warning(f"Failed to list models from {provider.name}: {e}")
        
        return all_models

    async def get_model(self, name: str) -> ModelInfo | None:
        """
        Get information about a specific model.
        
        Args:
            name: Model name or alias
            
        Returns:
            ModelInfo or None if not found
        """
        resolved_name = self.resolve_model(name)
        
        for provider in self._providers.values():
            try:
                models = await provider.list_models()
                for model in models:
                    if model.name == resolved_name:
                        return model
            except Exception:
                continue
        
        return None

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of all providers.
        
        Returns:
            Dictionary with health status per provider
        """
        results: dict[str, Any] = {}
        
        async def check_provider(name: str, provider: BaseProvider) -> tuple[str, bool]:
            try:
                healthy = await provider.health_check()
                return name, healthy
            except Exception:
                return name, False

        tasks = [
            check_provider(name, provider)
            for name, provider in self._providers.items()
        ]
        
        if tasks:
            check_results = await asyncio.gather(*tasks)
            for name, healthy in check_results:
                results[name] = "connected" if healthy else "disconnected"
        
        return results

    @property
    def providers(self) -> dict[str, BaseProvider]:
        """Get all registered providers."""
        return self._providers

    @property
    def aliases(self) -> dict[str, str]:
        """Get model aliases."""
        return self._aliases


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry