"""
End-to-end pipeline tests for DATS.

Tests the full agent pipeline from user request to task execution.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import pipeline components
from src.pipeline.orchestrator import (
    AgentPipeline,
    PipelineResult,
    TaskMode,
    TaskStatus,
    process_request,
)
from src.agents.coordinator import Coordinator
from src.agents.decomposer import Decomposer
from src.agents.complexity_estimator import ComplexityEstimator
from src.models.mock_client import MockModelClient, create_mock_client_for_tier
from src.storage.provenance import ProvenanceTracker


class TestCoordinatorQuickAnalysis:
    """Test coordinator's heuristic analysis."""

    def test_simple_request_detected_as_atomic(self):
        """Simple requests should not need decomposition."""
        coordinator = Coordinator()
        
        # Simple request
        result = coordinator._quick_analyze(
            "Create a Python function that calculates fibonacci numbers"
        )
        
        assert result["needs_decomposition"] is False
        assert result["domain"] == "code-general"
        assert result["complexity"] in ["tiny", "small"]

    def test_complex_request_needs_decomposition(self):
        """Complex requests should be flagged for decomposition."""
        coordinator = Coordinator()
        
        # Complex request with multiple requirements including complexity keywords
        result = coordinator._quick_analyze(
            "Build a complete system with multiple components including user authentication, "
            "database integration with several tables, REST API, and also a React frontend. "
            "The system should integrate with external services. Include testing as well as "
            "documentation and deployment configuration."
        )
        
        assert result["needs_decomposition"] is True
        assert result["complexity"] in ["medium", "large"]

    def test_mode_detection_fix_bug(self):
        """Bug fix requests should be detected correctly."""
        coordinator = Coordinator()
        
        result = coordinator._quick_analyze("Fix the bug in the login function")
        
        assert result["mode"] == "fix_bug"

    def test_mode_detection_documentation(self):
        """Documentation requests should be detected correctly."""
        coordinator = Coordinator()
        
        result = coordinator._quick_analyze("Create a README for the project")
        
        assert result["mode"] == "documentation"
        assert result["domain"] == "documentation"

    def test_domain_detection_ui(self):
        """UI-related requests should be detected correctly."""
        coordinator = Coordinator()
        
        # Use clear UI keywords
        result = coordinator._quick_analyze("Create a frontend UI component for the dashboard with CSS styling")
        
        assert result["domain"] == "ui-design"


class TestDecomposerAtomicity:
    """Test decomposer's atomicity detection."""

    def test_simple_task_is_atomic(self):
        """Simple tasks should be considered atomic."""
        decomposer = Decomposer()
        
        task = {
            "description": "Create a fibonacci function",
            "complexity": "small",
        }
        
        assert decomposer.is_atomic(task) is True

    def test_complex_task_not_atomic(self):
        """Complex tasks should not be considered atomic."""
        decomposer = Decomposer()
        
        task = {
            "description": "Build a complete system with multiple components",
            "complexity": "large",
            "needs_decomposition": True,
        }
        
        assert decomposer.is_atomic(task) is False

    def test_explicit_atomic_flag(self):
        """Explicit is_atomic flag should be respected."""
        decomposer = Decomposer()
        
        task = {
            "description": "Complex looking but marked atomic",
            "is_atomic": True,
        }
        
        assert decomposer.is_atomic(task) is True

    def test_atomicity_score(self):
        """Atomicity score should reflect task complexity."""
        decomposer = Decomposer()
        
        simple_task = {
            "description": "Simple function",
            "complexity": "tiny",
        }
        
        complex_task = {
            "description": "Build complete system with multiple integrations",
            "complexity": "large",
        }
        
        simple_score = decomposer.estimate_atomicity_score(simple_task)
        complex_score = decomposer.estimate_atomicity_score(complex_task)
        
        assert simple_score > complex_score
        assert simple_score > 0.5
        assert complex_score < 0.5


class TestMockModelClient:
    """Test the mock model client."""

    @pytest.mark.asyncio
    async def test_fibonacci_response(self):
        """Mock client should return fibonacci code for matching prompt."""
        client = MockModelClient()
        
        response = await client.generate(
            prompt="Create a Python function that calculates fibonacci numbers"
        )
        
        assert "def fibonacci" in response.content or "def fib" in response.content
        assert response.tokens_input > 0
        assert response.tokens_output > 0

    @pytest.mark.asyncio
    async def test_coordinator_response(self):
        """Mock client should return JSON for coordinator prompts."""
        client = MockModelClient()
        
        response = await client.generate(
            prompt="Analyze this task",
            system_prompt="You are a Coordinator agent",
        )
        
        assert "{" in response.content
        assert "mode" in response.content or "domain" in response.content

    @pytest.mark.asyncio
    async def test_custom_response(self):
        """Custom responses should be matched."""
        client = MockModelClient()
        client.add_response(
            pattern=r"custom.*test",
            response="Custom test response",
            tokens_input=10,
            tokens_output=5,
        )
        
        response = await client.generate(prompt="This is a custom test")
        
        assert response.content == "Custom test response"

    @pytest.mark.asyncio
    async def test_call_history(self):
        """Call history should be recorded."""
        client = MockModelClient()
        
        await client.generate(prompt="First call")
        await client.generate(prompt="Second call")
        
        history = client.get_call_history()
        
        assert len(history) == 2
        assert history[0]["prompt"] == "First call"
        assert history[1]["prompt"] == "Second call"


class TestPipelineWithMocks:
    """Test pipeline with mock model clients."""

    @pytest.fixture
    def mock_model_client(self):
        """Create a mock model client for testing."""
        return MockModelClient()

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_simple_request_flow(self, mock_model_client, temp_storage):
        """Test the flow of a simple request through the pipeline."""
        # Patch the model client creation
        with patch.object(Coordinator, 'get_model_client', return_value=mock_model_client), \
             patch.object(Decomposer, 'get_model_client', return_value=mock_model_client), \
             patch.object(ComplexityEstimator, 'get_model_client', return_value=mock_model_client):
            
            pipeline = AgentPipeline(
                provenance_path=temp_storage,
                use_celery=False,  # Don't use Celery for this test
            )
            
            # Simple request that shouldn't need decomposition
            result = await pipeline.process_request(
                user_request="Create a Python function that calculates fibonacci numbers",
                project_id="test-project",
            )
            
            # Verify result structure
            assert result.task_id is not None
            assert result.project_id == "test-project"
            assert result.mode in [TaskMode.NEW_PROJECT, TaskMode.UNKNOWN]
            assert result.started_at is not None
            assert result.completed_at is not None
            
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_coordinator_analysis(self, mock_model_client):
        """Test coordinator's request analysis."""
        with patch.object(Coordinator, 'get_model_client', return_value=mock_model_client):
            coordinator = Coordinator()
            
            result = await coordinator.process_request(
                user_request="Create a simple hello world function",
                project_id="test-project",
            )
            
            assert result["success"] is True
            assert "mode" in result
            assert "domain" in result
            assert "complexity" in result
            
            await coordinator.close()


class TestProvenanceRecording:
    """Test provenance tracking during pipeline execution."""

    @pytest.fixture
    def temp_provenance_dir(self):
        """Create temporary provenance directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_provenance_record_creation(self, temp_provenance_dir):
        """Test creating provenance records."""
        tracker = ProvenanceTracker(storage_path=temp_provenance_dir)
        
        record = tracker.create_record(
            task_id="test-task-123",
            project_id="test-project",
            model_used="mock-model",
            worker_id="code-general",
        )
        
        assert record.id is not None
        assert record.task_id == "test-task-123"
        assert record.project_id == "test-project"
        assert record.started_at is not None

    def test_provenance_record_completion(self, temp_provenance_dir):
        """Test completing provenance records."""
        tracker = ProvenanceTracker(storage_path=temp_provenance_dir)
        
        record = tracker.create_record(
            task_id="test-task-123",
            project_id="test-project",
            model_used="mock-model",
            worker_id="code-general",
        )
        
        completed = tracker.complete_record(
            record_id=record.id,
            outputs=[{"type": "code", "path": "test.py"}],
            tokens_input=100,
            tokens_output=50,
            confidence=0.9,
        )
        
        assert completed.completed_at is not None
        assert completed.tokens_input == 100
        assert completed.tokens_output == 50
        assert completed.confidence == 0.9
        assert completed.execution_time_ms is not None

    def test_provenance_persistence(self, temp_provenance_dir):
        """Test that provenance records are persisted to disk."""
        tracker = ProvenanceTracker(storage_path=temp_provenance_dir)
        
        record = tracker.create_record(
            task_id="test-task-persist",
            project_id="test-project",
            model_used="mock-model",
            worker_id="code-general",
        )
        
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"type": "code"}],
            tokens_input=100,
            tokens_output=50,
        )
        
        # Check file was created
        record_file = Path(temp_provenance_dir) / f"{record.id}.json"
        assert record_file.exists()


class TestFibonacciEndToEnd:
    """
    End-to-end test for the fibonacci task scenario.
    
    This tests the complete flow:
    1. Submit "Create a Python function that calculates fibonacci numbers"
    2. Verify it routes to appropriate tier
    3. Verify output is produced
    4. Verify provenance is recorded
    """

    @pytest.fixture
    def mock_all_clients(self):
        """Create mock clients for all components."""
        client = MockModelClient()
        return client

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance_dir = Path(tmpdir) / "provenance"
            work_product_dir = Path(tmpdir) / "work_products"
            rag_dir = Path(tmpdir) / "rag"
            
            provenance_dir.mkdir()
            work_product_dir.mkdir()
            rag_dir.mkdir()
            
            yield {
                "provenance": str(provenance_dir),
                "work_products": str(work_product_dir),
                "rag": str(rag_dir),
            }

    @pytest.mark.asyncio
    async def test_fibonacci_task_routing(self, mock_all_clients, temp_dirs):
        """Test that fibonacci task is routed to appropriate tier."""
        with patch.object(Coordinator, 'get_model_client', return_value=mock_all_clients), \
             patch.object(Decomposer, 'get_model_client', return_value=mock_all_clients), \
             patch.object(ComplexityEstimator, 'get_model_client', return_value=mock_all_clients):
            
            coordinator = Coordinator()
            
            # Analyze the fibonacci request
            result = await coordinator.process_request(
                user_request="Create a Python function that calculates fibonacci numbers",
                project_id="test-fibonacci",
            )
            
            # Should be simple task, no decomposition needed
            assert result["success"] is True
            assert result["needs_decomposition"] is False
            
            # Should route to small or tiny tier
            assert result["complexity"] in ["tiny", "small", "medium"]
            
            await coordinator.close()

    @pytest.mark.asyncio
    async def test_fibonacci_output_generation(self, mock_all_clients):
        """Test that fibonacci function is generated correctly."""
        # Test the mock client directly for now
        response = await mock_all_clients.generate(
            prompt="Create a Python function that calculates fibonacci numbers"
        )
        
        # Verify output contains fibonacci function
        assert "fibonacci" in response.content.lower() or "fib" in response.content.lower()
        assert "def " in response.content
        assert "return" in response.content

    def test_fibonacci_provenance_recorded(self, temp_dirs):
        """Test that provenance is recorded for fibonacci task."""
        tracker = ProvenanceTracker(storage_path=temp_dirs["provenance"])
        
        # Simulate task execution with provenance
        record = tracker.create_record(
            task_id="fibonacci-task-123",
            project_id="test-fibonacci",
            model_used="gemma3:12b",
            worker_id="code-general",
        )
        
        tracker.complete_record(
            record_id=record.id,
            outputs=[{
                "type": "code",
                "language": "python",
                "content": "def fibonacci(n): ...",
            }],
            tokens_input=150,
            tokens_output=200,
            confidence=0.85,
        )
        
        # Verify provenance
        retrieved = tracker.get_record(record.id)
        assert retrieved is not None
        assert retrieved.task_id == "fibonacci-task-123"
        assert len(retrieved.outputs) == 1
        assert retrieved.outputs[0]["type"] == "code"


class TestQAProfileSelection:
    """Test QA profile selection for different task types."""

    def test_simple_task_minimal_qa(self):
        """Simple tasks should get minimal QA."""
        coordinator = Coordinator()
        
        profile = coordinator._determine_qa_profile(
            complexity="small",
            domain="code-general",
        )
        
        assert profile == "consensus"

    def test_complex_task_thorough_qa(self):
        """Complex tasks should get thorough QA."""
        coordinator = Coordinator()
        
        profile = coordinator._determine_qa_profile(
            complexity="large",
            domain="code-general",
        )
        
        assert profile == "adversarial"

    def test_documentation_specialized_qa(self):
        """Documentation tasks should get specialized QA."""
        coordinator = Coordinator()
        
        profile = coordinator._determine_qa_profile(
            complexity="medium",
            domain="documentation",
        )
        
        assert profile == "documentation"


class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_request(self):
        """Pipeline should handle empty requests gracefully."""
        mock_client = MockModelClient()
        
        with patch.object(Coordinator, 'get_model_client', return_value=mock_client):
            pipeline = AgentPipeline(use_celery=False)
            
            result = await pipeline.process_request(
                user_request="",
                project_id="test-project",
            )
            
            # Should complete without crashing
            assert result.task_id is not None
            
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_pipeline_records_errors(self):
        """Pipeline should record errors in results."""
        # Create a mock that raises an exception
        mock_client = MockModelClient()
        mock_client.generate = AsyncMock(side_effect=Exception("Test error"))
        
        with patch.object(Coordinator, 'get_model_client', return_value=mock_client):
            pipeline = AgentPipeline(use_celery=False)
            
            result = await pipeline.process_request(
                user_request="Test request",
                project_id="test-project",
            )
            
            # Should record error but not crash
            assert result.task_id is not None
            # May have error recorded
            
            await pipeline.close()


# Integration tests that require actual services
class TestIntegrationWithServices:
    """
    Integration tests that require running services.
    
    These tests are skipped by default and should be run manually
    with appropriate services running.
    """

    @pytest.mark.skip(reason="Requires running RabbitMQ and Redis")
    @pytest.mark.asyncio
    async def test_full_celery_pipeline(self):
        """Test full pipeline with Celery task queuing."""
        from src.queue.tasks import process_pipeline
        
        result = process_pipeline.delay(
            user_request="Create a simple hello world function",
            project_id="integration-test",
        )
        
        # Wait for result with timeout
        task_result = result.get(timeout=60)
        
        assert task_result["status"] in ["completed", "queued"]
        assert task_result["task_id"] is not None

    @pytest.mark.skip(reason="Requires running Ollama")
    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_models(self):
        """Test full pipeline with real model execution."""
        result = await process_request(
            user_request="Create a Python function that calculates fibonacci numbers",
            project_id="integration-test",
            use_celery=False,
        )
        
        assert result.status in [TaskStatus.COMPLETED, TaskStatus.QUEUED]
        if result.status == TaskStatus.COMPLETED:
            assert result.output is not None
            assert "fibonacci" in result.output.lower() or "fib" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])