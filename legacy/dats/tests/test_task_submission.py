"""
Test task submission for DATS.

Tests that tasks can be submitted and executed (using eager mode).
"""

import pytest
import os


# Set up eager mode before importing celery app
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"
os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "True"
os.environ["ROUTING_CONFIG_PATH"] = "../prompts/schemas/routing_config.yaml"


class TestTaskSubmission:
    """Test task submission functionality."""

    def test_execute_task_small_tier(self):
        """Test executing a task on the small tier."""
        from src.queue.celery_app import app
        
        # Enable eager mode for testing
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_small
        
        task_data = {
            "id": "task-123",
            "project_id": "project-456",
            "type": "execute",
            "domain": "code-general",
            "description": "Implement a hello world function",
            "routing": {"tier": "small"},
        }
        
        # Execute task directly (eager mode)
        result = execute_small.apply(args=[task_data])
        
        assert result.successful()
        output = result.result
        
        assert output["task_id"] == "task-123"
        assert output["status"] == "completed"
        assert output["tier"] == "small"
        assert "output" in output

    def test_execute_task_tiny_tier(self):
        """Test executing a task on the tiny tier."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_tiny
        
        task_data = {
            "id": "task-tiny-001",
            "project_id": "project-789",
            "type": "execute",
            "domain": "code-general",
            "description": "Simple code formatting task",
        }
        
        result = execute_tiny.apply(args=[task_data])
        
        assert result.successful()
        output = result.result
        
        assert output["task_id"] == "task-tiny-001"
        assert output["tier"] == "tiny"

    def test_execute_task_large_tier(self):
        """Test executing a task on the large tier."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_large
        
        task_data = {
            "id": "task-large-001",
            "project_id": "project-abc",
            "type": "execute",
            "domain": "code-general",
            "description": "Complex refactoring task",
        }
        
        result = execute_large.apply(args=[task_data])
        
        assert result.successful()
        output = result.result
        
        assert output["task_id"] == "task-large-001"
        assert output["tier"] == "large"

    def test_execute_task_frontier_tier(self):
        """Test executing a task on the frontier tier."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_frontier
        
        task_data = {
            "id": "task-frontier-001",
            "project_id": "project-xyz",
            "type": "execute",
            "domain": "code-general",
            "description": "Complex architecture design",
        }
        
        result = execute_frontier.apply(args=[task_data])
        
        assert result.successful()
        output = result.result
        
        assert output["task_id"] == "task-frontier-001"
        assert output["tier"] == "frontier"

    def test_decompose_task(self):
        """Test task decomposition."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import decompose_task
        
        task_data = {
            "id": "parent-task-001",
            "project_id": "project-001",
            "type": "decompose",
            "description": "Build a REST API",
        }
        
        result = decompose_task.apply(args=[task_data])
        
        assert result.successful()
        output = result.result
        
        assert output["parent_task_id"] == "parent-task-001"
        assert output["status"] == "decomposed"
        assert "subtasks" in output

    def test_validate_task(self):
        """Test task validation."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import validate_task
        
        task_result = {
            "task_id": "task-to-validate",
            "output": {"content": "print('hello')"},
        }
        
        result = validate_task.apply(args=["task-to-validate", task_result])
        
        assert result.successful()
        output = result.result
        
        assert output["task_id"] == "task-to-validate"
        assert output["validation_status"] == "pending"

    def test_merge_results(self):
        """Test result merging."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import merge_results
        
        subtask_results = [
            {"task_id": "sub-1", "output": {"content": "code1"}},
            {"task_id": "sub-2", "output": {"content": "code2"}},
        ]
        
        result = merge_results.apply(args=["parent-001", subtask_results])
        
        assert result.successful()
        output = result.result
        
        assert output["parent_task_id"] == "parent-001"
        assert output["status"] == "merged"

    def test_task_output_contains_provenance(self):
        """Test that task output contains provenance information."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_small
        
        task_data = {
            "id": "provenance-test-001",
            "project_id": "project-prov",
            "type": "execute",
            "domain": "code-general",
            "description": "Test provenance tracking",
        }
        
        result = execute_small.apply(args=[task_data])
        output = result.result
        
        assert "provenance" in output
        provenance = output["provenance"]
        
        assert "model_used" in provenance
        assert "started_at" in provenance
        assert "completed_at" in provenance


class TestTaskDataValidation:
    """Test task data validation."""

    def test_task_with_minimal_data(self):
        """Test task with minimal required data."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_small
        
        # Minimal task data
        task_data = {
            "description": "Simple task",
        }
        
        result = execute_small.apply(args=[task_data])
        
        # Should still work with default values
        assert result.successful()
        output = result.result
        assert output["status"] == "completed"

    def test_task_with_custom_domain(self):
        """Test task with custom domain."""
        from src.queue.celery_app import app
        
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True
        
        from src.queue.tasks import execute_small
        
        task_data = {
            "id": "vision-task-001",
            "domain": "code-vision",
            "description": "ML code task",
        }
        
        result = execute_small.apply(args=[task_data])
        output = result.result
        
        assert output["domain"] == "code-vision"