# File: services/model-gateway/tests/unit/test_providers.py
"""Unit tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.providers.base import GenerateRequest, GenerateResponse, ModelInfo, StreamChunk
from src.providers.ollama import OllamaProvider
from src.providers.vllm import VLLMProvider


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        """Create an OllamaProvider for testing."""
        return OllamaProvider(
            name="test_ollama",
            endpoint="http://localhost:11434",
            timeout=30.0,
            configured_models=[
                {"name": "gemma3:12b", "tier": "small", "context_window": 32768},
            ],
        )

    def test_provider_type(self, provider: OllamaProvider):
        """Test provider type returns correct value."""
        assert provider.provider_type == "ollama"

    def test_provider_name(self, provider: OllamaProvider):
        """Test provider name is set correctly."""
        assert provider.name == "test_ollama"

    def test_provider_endpoint(self, provider: OllamaProvider):
        """Test provider endpoint is set correctly."""
        assert provider.endpoint == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_generate(self, provider: OllamaProvider, mock_ollama_response: dict):
        """Test generate method."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            request = GenerateRequest(
                model="gemma3:12b",
                prompt="Hello",
                temperature=0.7,
                max_tokens=100,
            )
            
            response = await provider.generate(request)
            
            assert isinstance(response, GenerateResponse)
            assert response.content == "Hello! I'm here to help."
            assert response.model == "gemma3:12b"
            assert response.tokens_input == 10
            assert response.tokens_output == 20

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider: OllamaProvider):
        """Test health check returns True when endpoint is healthy."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.health_check()
            
            assert result is True
            assert provider.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider: OllamaProvider):
        """Test health check returns False when endpoint is unhealthy."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_get_client.return_value = mock_client

            result = await provider.health_check()
            
            assert result is False
            assert provider.is_healthy is False


class TestVLLMProvider:
    """Tests for VLLMProvider."""

    @pytest.fixture
    def provider(self) -> VLLMProvider:
        """Create a VLLMProvider for testing."""
        return VLLMProvider(
            name="test_vllm",
            endpoint="http://localhost:8000/v1",
            timeout=30.0,
            configured_models=[
                {"name": "openai/gpt-oss-20b", "tier": "large", "context_window": 32768},
            ],
        )

    def test_provider_type(self, provider: VLLMProvider):
        """Test provider type returns correct value."""
        assert provider.provider_type == "vllm"

    def test_provider_name(self, provider: VLLMProvider):
        """Test provider name is set correctly."""
        assert provider.name == "test_vllm"

    @pytest.mark.asyncio
    async def test_generate(self, provider: VLLMProvider, mock_vllm_response: dict):
        """Test generate method with chat completions."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_vllm_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            request = GenerateRequest(
                model="openai/gpt-oss-20b",
                prompt="Hello",
                system_prompt="You are helpful.",
                temperature=0.7,
                max_tokens=100,
            )
            
            response = await provider.generate(request)
            
            assert isinstance(response, GenerateResponse)
            assert response.content == "Hello! I'm here to help."
            assert response.tokens_input == 10
            assert response.tokens_output == 20


class TestGenerateRequest:
    """Tests for GenerateRequest model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        request = GenerateRequest(model="test", prompt="Hello")
        
        assert request.temperature == 0.7
        assert request.max_tokens == 2000
        assert request.stream is False
        assert request.system_prompt is None
        assert request.stop_sequences is None

    def test_custom_values(self):
        """Test custom values are set correctly."""
        request = GenerateRequest(
            model="test",
            prompt="Hello",
            system_prompt="Be helpful",
            temperature=0.5,
            max_tokens=1000,
            stop_sequences=["STOP"],
            stream=True,
        )
        
        assert request.temperature == 0.5
        assert request.max_tokens == 1000
        assert request.stream is True
        assert request.system_prompt == "Be helpful"
        assert request.stop_sequences == ["STOP"]

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            GenerateRequest(model="test", prompt="Hello", temperature=-0.1)
        
        with pytest.raises(ValueError):
            GenerateRequest(model="test", prompt="Hello", temperature=2.5)


class TestGenerateResponse:
    """Tests for GenerateResponse model."""

    def test_total_tokens(self, sample_generate_response: GenerateResponse):
        """Test total_tokens property."""
        assert sample_generate_response.total_tokens == 150

    def test_default_id_generation(self):
        """Test that ID is generated if not provided."""
        response = GenerateResponse(
            model="test",
            provider="test",
            content="Hello",
        )
        assert response.id.startswith("gen-")