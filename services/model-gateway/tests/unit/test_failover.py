# File: services/model-gateway/tests/unit/test_failover.py
"""Unit tests for failover service."""

import pytest
from unittest.mock import patch

from src.providers.base import ModelInfo
from src.services.failover import FailoverService


class TestFailoverService:
    """Tests for FailoverService."""

    @pytest.fixture
    def service(self) -> FailoverService:
        """Create a FailoverService for testing."""
        service = FailoverService()
        service._strategies = ["same_tier", "tier_up", "provider_fallback"]
        service._tier_order = ["tiny", "small", "medium", "large", "frontier"]
        service._initialized = True
        return service

    @pytest.fixture
    def available_models(self) -> list[ModelInfo]:
        """Create a list of available models for testing."""
        return [
            ModelInfo(name="gemma3:4b", provider="ollama", endpoint_name="gpu", tier="tiny", status="available"),
            ModelInfo(name="gemma3:12b", provider="ollama", endpoint_name="gpu", tier="small", status="available"),
            ModelInfo(name="gemma3:27b", provider="ollama", endpoint_name="gpu", tier="medium", status="available"),
            ModelInfo(name="gpt-oss-20b", provider="vllm", endpoint_name="vllm", tier="large", status="available"),
            ModelInfo(name="claude-sonnet", provider="anthropic", endpoint_name="api", tier="frontier", status="available"),
        ]

    def test_get_tier_index(self, service: FailoverService):
        """Test tier index lookup."""
        assert service.get_tier_index("tiny") == 0
        assert service.get_tier_index("small") == 1
        assert service.get_tier_index("large") == 3
        assert service.get_tier_index("unknown") == -1

    def test_find_alternatives_same_tier(self, service: FailoverService, available_models: list[ModelInfo]):
        """Test finding alternatives in same tier."""
        # Add another small model from different provider
        available_models.append(
            ModelInfo(name="small-alt", provider="vllm", endpoint_name="vllm", tier="small", status="available")
        )
        
        alternatives = service.find_alternatives(
            failed_model="gemma3:12b",
            failed_provider="ollama",
            all_models=available_models,
            failed_tier="small",
        )
        
        # First alternative should be from different provider (same tier)
        assert len(alternatives) > 0
        assert alternatives[0].name == "small-alt"

    def test_find_alternatives_tier_up(self, service: FailoverService, available_models: list[ModelInfo]):
        """Test finding alternatives in higher tier."""
        alternatives = service.find_alternatives(
            failed_model="gemma3:4b",
            failed_provider="ollama",
            all_models=available_models,
            failed_tier="tiny",
        )
        
        # Should include higher tier models
        tiers = [m.tier for m in alternatives]
        assert "small" in tiers or "medium" in tiers or "large" in tiers

    def test_find_alternatives_provider_fallback(self, service: FailoverService, available_models: list[ModelInfo]):
        """Test finding alternatives from different provider."""
        alternatives = service.find_alternatives(
            failed_model="gemma3:12b",
            failed_provider="ollama",
            all_models=available_models,
            failed_tier="small",
        )
        
        # Should include models from other providers
        providers = set(m.provider for m in alternatives)
        assert "vllm" in providers or "anthropic" in providers

    def test_find_alternatives_no_available(self, service: FailoverService):
        """Test finding alternatives when none available."""
        unavailable_models = [
            ModelInfo(name="model1", provider="test", endpoint_name="test", tier="small", status="unavailable"),
        ]
        
        alternatives = service.find_alternatives(
            failed_model="model2",
            failed_provider="test",
            all_models=unavailable_models,
        )
        
        assert len(alternatives) == 0

    def test_should_retry_connection_error(self, service: FailoverService):
        """Test retry decision for connection errors."""
        error = ConnectionError("Connection refused")
        
        assert service.should_retry(error, attempt=1) is True
        assert service.should_retry(error, attempt=2) is True
        assert service.should_retry(error, attempt=3) is False  # Max attempts

    def test_should_retry_timeout_error(self, service: FailoverService):
        """Test retry decision for timeout errors."""
        error = TimeoutError("Request timed out")
        
        assert service.should_retry(error, attempt=1) is True

    def test_should_retry_value_error(self, service: FailoverService):
        """Test retry decision for non-retryable errors."""
        error = ValueError("Invalid model")
        
        assert service.should_retry(error, attempt=1) is False

    def test_should_retry_max_attempts(self, service: FailoverService):
        """Test retry stops at max attempts."""
        error = ConnectionError("Connection refused")
        
        assert service.should_retry(error, attempt=3, max_attempts=3) is False
        assert service.should_retry(error, attempt=2, max_attempts=3) is True