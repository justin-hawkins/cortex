# File: services/model-gateway/tests/contract/test_openapi.py
"""Contract tests validating API against OpenAPI specification."""

import pytest
from pathlib import Path

import yaml


class TestOpenAPIContract:
    """Tests to validate OpenAPI contract specification."""

    @pytest.fixture
    def openapi_spec(self) -> dict:
        """Load the OpenAPI specification."""
        spec_path = Path(__file__).parent.parent.parent / "contracts" / "openapi.yaml"
        with open(spec_path) as f:
            return yaml.safe_load(f)

    def test_spec_has_required_info(self, openapi_spec: dict):
        """Test OpenAPI spec has required info fields."""
        assert "openapi" in openapi_spec
        assert openapi_spec["openapi"].startswith("3.")
        
        assert "info" in openapi_spec
        info = openapi_spec["info"]
        assert "title" in info
        assert "version" in info
        assert info["version"] == "1.0.0"

    def test_spec_has_required_paths(self, openapi_spec: dict):
        """Test OpenAPI spec defines required paths."""
        assert "paths" in openapi_spec
        paths = openapi_spec["paths"]
        
        # Health endpoints
        assert "/health" in paths
        assert "/health/detailed" in paths
        
        # Models endpoints
        assert "/models" in paths
        assert "/models/{name}" in paths
        
        # Generate endpoint
        assert "/generate" in paths

    def test_health_endpoint_spec(self, openapi_spec: dict):
        """Test health endpoint specification."""
        health = openapi_spec["paths"]["/health"]
        
        assert "get" in health
        get_op = health["get"]
        assert "operationId" in get_op
        assert "responses" in get_op
        assert "200" in get_op["responses"]

    def test_generate_endpoint_spec(self, openapi_spec: dict):
        """Test generate endpoint specification."""
        generate = openapi_spec["paths"]["/generate"]
        
        assert "post" in generate
        post_op = generate["post"]
        assert "operationId" in post_op
        assert "requestBody" in post_op
        assert "responses" in post_op
        
        # Check required responses
        assert "200" in post_op["responses"]
        assert "400" in post_op["responses"]
        assert "404" in post_op["responses"]

    def test_schemas_defined(self, openapi_spec: dict):
        """Test required schemas are defined."""
        assert "components" in openapi_spec
        assert "schemas" in openapi_spec["components"]
        schemas = openapi_spec["components"]["schemas"]
        
        required_schemas = [
            "HealthResponse",
            "DetailedHealthResponse",
            "ModelResponse",
            "ModelsListResponse",
            "GenerateRequest",
            "GenerateResponse",
            "ErrorResponse",
        ]
        
        for schema_name in required_schemas:
            assert schema_name in schemas, f"Missing schema: {schema_name}"

    def test_generate_request_schema(self, openapi_spec: dict):
        """Test GenerateRequest schema has required fields."""
        schemas = openapi_spec["components"]["schemas"]
        gen_request = schemas["GenerateRequest"]
        
        assert "required" in gen_request
        assert "model" in gen_request["required"]
        assert "prompt" in gen_request["required"]
        
        props = gen_request["properties"]
        assert "model" in props
        assert "prompt" in props
        assert "temperature" in props
        assert "max_tokens" in props
        assert "stream" in props

    def test_generate_response_schema(self, openapi_spec: dict):
        """Test GenerateResponse schema has required fields."""
        schemas = openapi_spec["components"]["schemas"]
        gen_response = schemas["GenerateResponse"]
        
        assert "required" in gen_response
        required_fields = gen_response["required"]
        
        expected_fields = [
            "id", "model", "provider", "content",
            "tokens_input", "tokens_output", "latency_ms",
            "finish_reason", "created_at", "metadata"
        ]
        
        for field in expected_fields:
            assert field in required_fields, f"Missing required field: {field}"

    def test_model_response_schema(self, openapi_spec: dict):
        """Test ModelResponse schema has required fields."""
        schemas = openapi_spec["components"]["schemas"]
        model_response = schemas["ModelResponse"]
        
        assert "properties" in model_response
        props = model_response["properties"]
        
        assert "name" in props
        assert "provider" in props
        assert "tier" in props
        assert "status" in props
        assert "context_window" in props

    def test_servers_defined(self, openapi_spec: dict):
        """Test servers are defined in spec."""
        assert "servers" in openapi_spec
        servers = openapi_spec["servers"]
        
        assert len(servers) >= 1
        for server in servers:
            assert "url" in server

    def test_tags_defined(self, openapi_spec: dict):
        """Test tags are defined for API organization."""
        assert "tags" in openapi_spec
        tags = openapi_spec["tags"]
        
        tag_names = [t["name"] for t in tags]
        assert "health" in tag_names
        assert "models" in tag_names
        assert "generate" in tag_names