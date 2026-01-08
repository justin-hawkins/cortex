"""
Pytest configuration and fixtures for DATS tests.

Sets up proper paths and mock configurations for testing.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_mock_routing_config():
    """Create a mock routing configuration for testing."""
    from src.config.routing import RoutingConfig, ModelTier, ModelConfig, EmbeddingConfig, AgentRouting
    
    return RoutingConfig(
        model_tiers={
            "tiny": ModelTier(
                context_window=32000,
                safe_working_limit=22000,
                models=[
                    ModelConfig(
                        name="mock-gemma-4b",
                        endpoint="mock://localhost",
                        type="ollama",
                        priority=1,
                    )
                ],
            ),
            "small": ModelTier(
                context_window=32000,
                safe_working_limit=22000,
                models=[
                    ModelConfig(
                        name="mock-gemma-12b",
                        endpoint="mock://localhost",
                        type="ollama",
                        priority=1,
                    )
                ],
            ),
            "large": ModelTier(
                context_window=64000,
                safe_working_limit=45000,
                models=[
                    ModelConfig(
                        name="mock-qwen-coder",
                        endpoint="mock://localhost",
                        type="openai_compatible",
                        priority=1,
                    )
                ],
            ),
            "frontier": ModelTier(
                context_window=200000,
                safe_working_limit=150000,
                models=[
                    ModelConfig(
                        name="mock-claude-sonnet",
                        endpoint="mock://localhost",
                        type="anthropic",
                        priority=1,
                    )
                ],
            ),
        },
        embedding=EmbeddingConfig(
            model="mock-embed",
            endpoint="mock://localhost",
            type="ollama",
        ),
        agent_routing={
            "coordinator": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "decomposer": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "complexity_estimator": AgentRouting(
                preferred_tier="small",
                preferred_model=None,
                fallback_tier="tiny",
            ),
            "qa_reviewer": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "merge_coordinator": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
        },
    )


# Create the mock config at module load time
_mock_routing_config = None


def get_mock_routing_config():
    """Get or create the mock routing config singleton."""
    global _mock_routing_config
    if _mock_routing_config is None:
        _mock_routing_config = create_mock_routing_config()
    return _mock_routing_config


# Patch routing config at import time
import src.config.routing as routing_module
routing_module._routing_config = get_mock_routing_config()
original_get_routing_config = routing_module.get_routing_config


def patched_get_routing_config(config_path=None):
    """Return mock config instead of loading from file."""
    return get_mock_routing_config()


routing_module.get_routing_config = patched_get_routing_config


@pytest.fixture(autouse=True)
def mock_routing_config():
    """
    Provide the mock routing configuration for tests.
    
    The config is already patched at module load time,
    this fixture just provides access to it.
    """
    yield get_mock_routing_config()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.config.settings import Settings
    
    return Settings(
        redis_host="localhost",
        redis_port=6379,
        rabbitmq_host="localhost",
        rabbitmq_port=5672,
        use_mock_models=True,
        provenance_path="data/test_provenance",
        work_product_path="data/test_work_products",
        rag_storage_path="data/test_rag",
    )


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
