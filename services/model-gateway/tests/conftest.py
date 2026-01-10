# File: services/model-gateway/tests/conftest.py
"""Pytest configuration and fixtures for Model Gateway tests."""

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app
from src.providers.base import GenerateResponse, ModelInfo, StreamChunk
from src.services.model_registry import ModelRegistry, get_registry


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock model registry."""
    registry = MagicMock(spec=ModelRegistry)
    registry.aliases = {
        "tiny": "gemma3:4b",
        "small": "gemma3:12b",
        "large": "openai/gpt-oss-20b",
    }
    return registry


@pytest.fixture
def sample_model_info() -> ModelInfo:
    """Create a sample ModelInfo for testing."""
    return ModelInfo(
        name="gemma3:12b",
        provider="ollama",
        endpoint_name="ollama_gpu_general",
        context_window=32768,
        tier="small",
        model_type="chat",
        description="Gemma 3 12B - balanced",
        status="available",
        metadata={"size": 12000000000},
    )


@pytest.fixture
def sample_generate_response() -> GenerateResponse:
    """Create a sample GenerateResponse for testing."""
    return GenerateResponse(
        id="gen-test-12345",
        model="gemma3:12b",
        provider="ollama_gpu_general",
        content="def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        tokens_input=50,
        tokens_output=100,
        latency_ms=1250,
        finish_reason="stop",
        metadata={"endpoint": "http://192.168.1.12:11434"},
    )


@pytest.fixture
def sample_stream_chunks() -> list[StreamChunk]:
    """Create sample StreamChunks for testing."""
    return [
        StreamChunk(id="gen-test-12345", content="def ", done=False),
        StreamChunk(id="gen-test-12345", content="fibonacci", done=False),
        StreamChunk(id="gen-test-12345", content="(n):", done=False),
        StreamChunk(id="gen-test-12345", content="", done=True, finish_reason="stop", tokens_output=10),
    ]


@pytest.fixture
def mock_ollama_response() -> dict:
    """Create a mock Ollama API response."""
    return {
        "model": "gemma3:12b",
        "response": "Hello! I'm here to help.",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


@pytest.fixture
def mock_vllm_response() -> dict:
    """Create a mock vLLM (OpenAI-compatible) API response."""
    return {
        "id": "cmpl-12345",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm here to help.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }