"""
Basic import tests for DATS modules.

Verifies that all modules can be imported without errors.
"""

import pytest


def test_import_config():
    """Test config module imports."""
    from src.config.settings import Settings, get_settings
    from src.config.routing import RoutingConfig, load_routing_config
    
    assert Settings is not None
    assert get_settings is not None
    assert RoutingConfig is not None
    assert load_routing_config is not None


def test_import_models():
    """Test models module imports."""
    from src.models.base import BaseModelClient, ModelResponse
    from src.models.ollama_client import OllamaClient
    from src.models.openai_client import OpenAICompatibleClient
    from src.models.anthropic_client import AnthropicClient
    
    assert BaseModelClient is not None
    assert ModelResponse is not None
    assert OllamaClient is not None
    assert OpenAICompatibleClient is not None
    assert AnthropicClient is not None


def test_import_prompts():
    """Test prompts module imports."""
    from src.prompts.loader import PromptLoader, LoadedPrompt
    from src.prompts.renderer import PromptRenderer
    
    assert PromptLoader is not None
    assert LoadedPrompt is not None
    assert PromptRenderer is not None


def test_import_queue():
    """Test queue module imports."""
    from src.queue.celery_app import app
    from src.queue.tasks import (
        execute_task,
        execute_tiny,
        execute_small,
        execute_large,
        execute_frontier,
    )
    
    assert app is not None
    assert execute_task is not None
    assert execute_tiny is not None
    assert execute_small is not None
    assert execute_large is not None
    assert execute_frontier is not None


def test_import_agents():
    """Test agents module imports."""
    from src.agents.base import BaseAgent, AgentContext, AgentResult
    from src.agents.coordinator import Coordinator
    from src.agents.decomposer import Decomposer
    from src.agents.complexity_estimator import ComplexityEstimator
    from src.agents.qa_reviewer import QAReviewer
    from src.agents.merge_coordinator import MergeCoordinator
    
    assert BaseAgent is not None
    assert AgentContext is not None
    assert AgentResult is not None
    assert Coordinator is not None
    assert Decomposer is not None
    assert ComplexityEstimator is not None
    assert QAReviewer is not None
    assert MergeCoordinator is not None


def test_import_workers():
    """Test workers module imports."""
    from src.workers.base import BaseWorker, WorkerContext, WorkerResult
    from src.workers.code_general import CodeGeneralWorker
    from src.workers.code_vision import CodeVisionWorker
    from src.workers.code_embedded import CodeEmbeddedWorker
    from src.workers.documentation import DocumentationWorker
    from src.workers.ui_design import UIDesignWorker
    
    assert BaseWorker is not None
    assert WorkerContext is not None
    assert WorkerResult is not None
    assert CodeGeneralWorker is not None
    assert CodeVisionWorker is not None
    assert CodeEmbeddedWorker is not None
    assert DocumentationWorker is not None
    assert UIDesignWorker is not None


def test_import_storage():
    """Test storage module imports."""
    from src.storage.provenance import ProvenanceRecord, ProvenanceTracker
    from src.storage.work_product import WorkProductStore, Artifact
    
    assert ProvenanceRecord is not None
    assert ProvenanceTracker is not None
    assert WorkProductStore is not None
    assert Artifact is not None


def test_model_response_dataclass():
    """Test ModelResponse dataclass functionality."""
    from src.models.base import ModelResponse
    
    response = ModelResponse(
        content="test content",
        tokens_input=100,
        tokens_output=50,
        model="test-model",
        finish_reason="stop",
    )
    
    assert response.content == "test content"
    assert response.total_tokens == 150


def test_provenance_record():
    """Test ProvenanceRecord functionality."""
    from src.storage.provenance import ProvenanceRecord
    
    record = ProvenanceRecord(
        task_id="test-task",
        project_id="test-project",
        model_used="test-model",
        worker_id="test-worker",
    )
    
    assert record.task_id == "test-task"
    assert record.id is not None  # Auto-generated


def test_artifact():
    """Test Artifact functionality."""
    from src.storage.work_product import Artifact
    
    artifact = Artifact(
        type="code",
        content="print('hello')",
        language="python",
    )
    
    assert artifact.type == "code"
    assert artifact.checksum is not None  # Auto-computed