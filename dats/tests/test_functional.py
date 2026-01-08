"""
Functional tests for DATS core components.

Tests actual functionality beyond simple imports.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestPromptLoader:
    """Test prompt loading functionality."""

    def test_loader_initialization(self):
        """Test PromptLoader can be initialized."""
        from src.prompts.loader import PromptLoader

        loader = PromptLoader(prompts_dir="../prompts")
        assert loader is not None
        assert loader.prompts_dir.name == "prompts"

    def test_load_existing_prompt(self):
        """Test loading an existing prompt file."""
        from src.prompts.loader import PromptLoader

        loader = PromptLoader(prompts_dir="../prompts")
        
        # Load the coordinator prompt
        prompt = loader.get("agents", "coordinator")
        
        assert prompt.content is not None
        assert len(prompt.content) > 0
        assert prompt.version_hash is not None
        assert len(prompt.version_hash) == 12  # SHA256[:12]

    def test_load_nonexistent_prompt_raises(self):
        """Test that loading a nonexistent prompt raises error."""
        from src.prompts.loader import PromptLoader

        loader = PromptLoader(prompts_dir="../prompts")
        
        with pytest.raises(FileNotFoundError):
            loader.get("agents", "nonexistent_agent")


class TestPromptRenderer:
    """Test prompt rendering functionality."""

    def test_renderer_initialization(self):
        """Test PromptRenderer can be initialized."""
        from src.prompts.renderer import PromptRenderer

        renderer = PromptRenderer()
        assert renderer is not None

    def test_render_with_variables(self):
        """Test rendering a template with variables."""
        from src.prompts.renderer import PromptRenderer

        renderer = PromptRenderer()
        
        # Test with a raw template string
        template = "Task {task_id}: {task_description} (Model: {model_name})"
        rendered = renderer.render(template, {
            "task_id": "test-123",
            "task_description": "Test task",
            "model_name": "test-model",
        })
        
        assert rendered == "Task test-123: Test task (Model: test-model)"
        assert "test-123" in rendered
        assert "test-model" in rendered


class TestModelResponse:
    """Test ModelResponse dataclass."""

    def test_model_response_creation(self):
        """Test creating a ModelResponse."""
        from src.models.base import ModelResponse

        response = ModelResponse(
            content="Hello, world!",
            tokens_input=10,
            tokens_output=5,
            model="test-model",
            finish_reason="stop",
        )

        assert response.content == "Hello, world!"
        assert response.tokens_input == 10
        assert response.tokens_output == 5
        assert response.total_tokens == 15

    def test_model_response_total_tokens(self):
        """Test ModelResponse total_tokens property."""
        from src.models.base import ModelResponse

        response = ModelResponse(
            content="Test",
            tokens_input=100,
            tokens_output=50,
            model="claude-3",
            finish_reason="stop",
        )

        assert response.total_tokens == 150
        assert response.model == "claude-3"


class TestProvenanceTracking:
    """Test provenance tracking functionality."""

    def test_create_provenance_record(self):
        """Test creating a provenance record."""
        from src.storage.provenance import ProvenanceRecord, ProvenanceTracker

        tracker = ProvenanceTracker()
        
        record = tracker.create_record(
            task_id="task-123",
            project_id="project-456",
            model_used="test-model",
            worker_id="worker-1",
        )

        assert record.id is not None
        assert record.task_id == "task-123"
        assert record.project_id == "project-456"
        assert record.started_at is not None

    def test_complete_provenance_record(self):
        """Test completing a provenance record."""
        from src.storage.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        
        record = tracker.create_record(
            task_id="task-123",
            project_id="project-456",
            model_used="test-model",
            worker_id="worker-1",
        )

        completed = tracker.complete_record(
            record_id=record.id,
            outputs=[{"type": "code", "path": "test.py"}],
            tokens_input=100,
            tokens_output=50,
            confidence=0.95,
        )

        assert completed.completed_at is not None
        assert completed.tokens_input == 100
        assert completed.tokens_output == 50
        assert completed.confidence == 0.95
        assert completed.execution_time_ms is not None

    def test_provenance_serialization(self):
        """Test provenance record serialization."""
        from src.storage.provenance import ProvenanceRecord

        record = ProvenanceRecord(
            task_id="task-123",
            project_id="project-456",
            model_used="test-model",
            worker_id="worker-1",
        )

        data = record.to_dict()
        restored = ProvenanceRecord.from_dict(data)

        assert restored.task_id == record.task_id
        assert restored.project_id == record.project_id


class TestWorkProductStore:
    """Test work product storage functionality."""

    def test_store_and_retrieve_artifact(self):
        """Test storing and retrieving an artifact."""
        from src.storage.work_product import WorkProductStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkProductStore(base_path=tmpdir)
            
            # Store an artifact
            artifact = store.store(
                content="print('hello')",
                artifact_type="code",
                language="python",
            )

            assert artifact.id is not None
            assert artifact.checksum is not None

            # Retrieve the artifact
            retrieved = store.get(artifact.id)
            assert retrieved is not None
            assert retrieved.content == "print('hello')"

    def test_list_artifacts(self):
        """Test listing artifacts."""
        from src.storage.work_product import WorkProductStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkProductStore(base_path=tmpdir)
            
            # Store multiple artifacts
            store.store(content="code1", artifact_type="code", language="python")
            store.store(content="code2", artifact_type="code", language="python")
            store.store(content="doc1", artifact_type="document", language="markdown")

            # List all
            all_artifacts = store.list_artifacts()
            assert len(all_artifacts) == 3

            # List by type
            code_artifacts = store.list_artifacts(artifact_type="code")
            assert len(code_artifacts) == 2

    def test_update_artifact(self):
        """Test updating an artifact."""
        from src.storage.work_product import WorkProductStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkProductStore(base_path=tmpdir)
            
            artifact = store.store(
                content="original",
                artifact_type="code",
            )
            original_checksum = artifact.checksum

            updated = store.update(artifact.id, content="modified")
            
            assert updated.content == "modified"
            assert updated.checksum != original_checksum

    def test_delete_artifact(self):
        """Test deleting an artifact."""
        from src.storage.work_product import WorkProductStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = WorkProductStore(base_path=tmpdir)
            
            artifact = store.store(content="to delete", artifact_type="code")
            artifact_id = artifact.id

            result = store.delete(artifact_id)
            assert result is True

            retrieved = store.get(artifact_id)
            assert retrieved is None


class TestRoutingConfig:
    """Test routing configuration."""

    def test_load_routing_config(self):
        """Test loading routing configuration."""
        from src.config.routing import load_routing_config

        config = load_routing_config("../prompts/schemas/routing_config.yaml")
        
        assert config is not None
        assert len(config.model_tiers) > 0

    def test_get_tier(self):
        """Test getting a specific tier."""
        from src.config.routing import load_routing_config

        config = load_routing_config("../prompts/schemas/routing_config.yaml")
        
        small_tier = config.get_tier("small")
        assert small_tier is not None
        assert small_tier.context_window > 0

    def test_get_model_for_agent(self):
        """Test getting model config for an agent."""
        from src.config.routing import load_routing_config

        config = load_routing_config("../prompts/schemas/routing_config.yaml")
        
        model = config.get_model_for_agent("coordinator")
        assert model is not None


class TestSettings:
    """Test settings configuration."""

    def test_settings_defaults(self):
        """Test settings have sensible defaults."""
        from src.config.settings import Settings

        settings = Settings()
        
        assert settings.redis_host is not None
        assert settings.redis_port > 0
        assert settings.worker_concurrency > 0

    def test_broker_url_construction(self):
        """Test broker URL is constructed correctly."""
        from src.config.settings import Settings

        settings = Settings(
            rabbitmq_host="localhost",
            rabbitmq_port=5672,
            rabbitmq_user="guest",
            rabbitmq_password="guest",
        )

        assert "amqp://" in settings.broker_url
        assert "localhost" in settings.broker_url
        assert "localhost" in settings.broker_url
