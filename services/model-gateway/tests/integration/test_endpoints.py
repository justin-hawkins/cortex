# File: services/model-gateway/tests/integration/test_endpoints.py
"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import app
from src.providers.base import GenerateResponse, ModelInfo


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, test_client: TestClient):
        """Test basic health check returns 200."""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns service info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "model-gateway"
        assert "version" in data
        assert "docs" in data


class TestModelsEndpoints:
    """Tests for models endpoints."""

    @pytest.fixture
    def mock_registry_with_models(self):
        """Create a mock registry with models."""
        with patch("src.routers.models.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.aliases = {"small": "gemma3:12b"}
            mock_registry.list_models = AsyncMock(return_value=[
                ModelInfo(
                    name="gemma3:12b",
                    provider="ollama",
                    endpoint_name="gpu",
                    tier="small",
                    status="available",
                ),
            ])
            mock_registry.get_model = AsyncMock(return_value=ModelInfo(
                name="gemma3:12b",
                provider="ollama",
                endpoint_name="gpu",
                tier="small",
                status="available",
            ))
            mock_get_registry.return_value = mock_registry
            yield mock_registry

    def test_list_models(self, test_client: TestClient, mock_registry_with_models):
        """Test listing models returns correct structure."""
        response = test_client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert "aliases" in data

    def test_get_model_not_found(self, test_client: TestClient):
        """Test getting non-existent model returns 404."""
        with patch("src.routers.models.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get_model = AsyncMock(return_value=None)
            mock_get_registry.return_value = mock_registry
            
            response = test_client.get("/api/v1/models/nonexistent-model")
            
            assert response.status_code == 404


class TestGenerateEndpoints:
    """Tests for generate endpoints."""

    @pytest.fixture
    def mock_registry_for_generate(self, sample_generate_response: GenerateResponse):
        """Create a mock registry for generation."""
        with patch("src.routers.generate.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.generate = AsyncMock(return_value=sample_generate_response)
            mock_get_registry.return_value = mock_registry
            yield mock_registry

    def test_generate_success(self, test_client: TestClient, mock_registry_for_generate):
        """Test successful generation."""
        response = test_client.post(
            "/api/v1/generate",
            json={
                "model": "gemma3:12b",
                "prompt": "Hello",
                "temperature": 0.7,
                "max_tokens": 100,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "model" in data
        assert "provider" in data

    def test_generate_missing_model(self, test_client: TestClient):
        """Test generation without model returns 422."""
        response = test_client.post(
            "/api/v1/generate",
            json={
                "prompt": "Hello",
            },
        )
        
        assert response.status_code == 422

    def test_generate_missing_prompt(self, test_client: TestClient):
        """Test generation without prompt returns 422."""
        response = test_client.post(
            "/api/v1/generate",
            json={
                "model": "gemma3:12b",
            },
        )
        
        assert response.status_code == 422

    def test_generate_invalid_temperature(self, test_client: TestClient):
        """Test generation with invalid temperature returns 422."""
        response = test_client.post(
            "/api/v1/generate",
            json={
                "model": "gemma3:12b",
                "prompt": "Hello",
                "temperature": 3.0,  # Invalid: > 2.0
            },
        )
        
        assert response.status_code == 422

    def test_generate_model_not_found(self, test_client: TestClient):
        """Test generation with unknown model returns 404."""
        with patch("src.routers.generate.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.generate = AsyncMock(
                side_effect=ValueError("Model not found: unknown-model")
            )
            mock_get_registry.return_value = mock_registry
            
            response = test_client.post(
                "/api/v1/generate",
                json={
                    "model": "unknown-model",
                    "prompt": "Hello",
                },
            )
            
            assert response.status_code == 404


class TestOpenAPISpec:
    """Tests for OpenAPI specification."""

    def test_openapi_available(self, test_client: TestClient):
        """Test OpenAPI spec is available."""
        response = test_client.get("/api/v1/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_docs_available(self, test_client: TestClient):
        """Test Swagger UI is available."""
        response = test_client.get("/api/v1/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]