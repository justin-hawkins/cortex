"""Tests for conflict detection."""

import pytest
from datetime import datetime

from src.merge.config import MergeConfig
from src.merge.detector import ConflictDetector, TaskOutput, InFlightTask
from src.merge.models import ConflictType
from src.storage.work_product import Artifact


class TestConflictDetector:
    """Tests for ConflictDetector."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MergeConfig()

    @pytest.fixture
    def detector(self, config):
        """Create test detector."""
        return ConflictDetector(config=config)

    def test_no_conflict_different_files(self, detector):
        """Two tasks modifying different files should not conflict."""
        task1 = TaskOutput(
            task_id="task-1",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-1",
                    path="src/module_a.py",
                    type="code",
                    content="def func_a(): pass",
                )
            ],
        )

        task2 = TaskOutput(
            task_id="task-2",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-2",
                    path="src/module_b.py",
                    type="code",
                    content="def func_b(): pass",
                )
            ],
        )

        # Check task1 (no prior completions)
        conflicts1 = detector.check_on_task_complete(task1)
        assert len(conflicts1) == 0

        # Check task2 (task1 already completed)
        conflicts2 = detector.check_on_task_complete(task2)
        assert len(conflicts2) == 0

    def test_conflict_same_file(self, detector):
        """Two tasks modifying same file should conflict."""
        task1 = TaskOutput(
            task_id="task-1",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-1",
                    path="src/module.py",
                    type="code",
                    content="def func_a():\n    return 1\n",
                )
            ],
        )

        task2 = TaskOutput(
            task_id="task-2",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-2",
                    path="src/module.py",
                    type="code",
                    content="def func_a():\n    return 2\n",
                )
            ],
        )

        # Check task1
        conflicts1 = detector.check_on_task_complete(task1)
        assert len(conflicts1) == 0

        # Check task2 - should find conflict with task1
        conflicts2 = detector.check_on_task_complete(task2)
        assert len(conflicts2) == 1

        conflict = conflicts2[0]
        assert conflict.project_id == "proj-1"
        assert len(conflict.involved_tasks) == 2
        assert len(conflict.affected_artifacts) == 1
        assert conflict.affected_artifacts[0].location == "src/module.py"

    def test_conflict_different_projects(self, detector):
        """Tasks in different projects should not conflict."""
        task1 = TaskOutput(
            task_id="task-1",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-1",
                    path="src/module.py",
                    type="code",
                    content="# Project 1",
                )
            ],
        )

        task2 = TaskOutput(
            task_id="task-2",
            project_id="proj-2",  # Different project
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-2",
                    path="src/module.py",  # Same path but different project
                    type="code",
                    content="# Project 2",
                )
            ],
        )

        detector.check_on_task_complete(task1)
        conflicts = detector.check_on_task_complete(task2)
        assert len(conflicts) == 0

    def test_in_flight_registration(self, detector):
        """Test in-flight task registration."""
        task = InFlightTask(
            task_id="task-1",
            project_id="proj-1",
            started_at=datetime.utcnow(),
            target_artifacts=["src/module.py"],
        )

        detector.register_in_flight(task)
        assert "task-1" in detector._in_flight

        detector.unregister_in_flight("task-1")
        assert "task-1" not in detector._in_flight

    def test_in_flight_overlap_warning(self, detector):
        """Test in-flight overlap detection."""
        # Register in-flight task
        in_flight = InFlightTask(
            task_id="task-inflight",
            project_id="proj-1",
            started_at=datetime.utcnow(),
            target_artifacts=["src/module.py"],
        )
        detector.register_in_flight(in_flight)

        # Complete a task that touches same file
        completed = TaskOutput(
            task_id="task-completed",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[
                Artifact(
                    id="art-1",
                    path="src/module.py",
                    type="code",
                    content="# Completed",
                )
            ],
        )

        # Should detect overlap with in-flight
        conflicts = detector.check_on_task_complete(completed)
        # No conflict record yet (task not complete), but logged warning


class TestConflictRegionComputation:
    """Tests for conflict region computation."""

    @pytest.fixture
    def detector(self):
        """Create test detector."""
        return ConflictDetector()

    def test_replace_conflict(self, detector):
        """Test detection of replace conflict."""
        ours = "line1\nmodified by A\nline3\n"
        theirs = "line1\nmodified by B\nline3\n"

        regions = detector._compute_conflict_regions(ours, theirs)

        assert len(regions) == 1
        assert "modified by A" in regions[0].ours_content
        assert "modified by B" in regions[0].theirs_content

    def test_no_conflict_identical(self, detector):
        """Identical content should have no conflict regions."""
        content = "line1\nline2\nline3\n"

        regions = detector._compute_conflict_regions(content, content)
        assert len(regions) == 0

    def test_multiple_conflicts(self, detector):
        """Test detection of multiple conflict regions."""
        ours = "line1\nA1\nline3\nA2\nline5\n"
        theirs = "line1\nB1\nline3\nB2\nline5\n"

        regions = detector._compute_conflict_regions(ours, theirs)

        # Should have 2 conflict regions
        assert len(regions) == 2


class TestTaskOutput:
    """Tests for TaskOutput dataclass."""

    def test_task_output_creation(self):
        """Test TaskOutput creation."""
        task = TaskOutput(
            task_id="task-1",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
        )

        assert task.task_id == "task-1"
        assert task.project_id == "proj-1"
        assert task.artifacts == []

    def test_task_output_with_artifacts(self):
        """Test TaskOutput with artifacts."""
        artifact = Artifact(
            id="art-1",
            path="src/test.py",
            type="code",
            content="# test",
        )

        task = TaskOutput(
            task_id="task-1",
            project_id="proj-1",
            completed_at=datetime.utcnow(),
            artifacts=[artifact],
        )

        assert len(task.artifacts) == 1
        assert task.artifacts[0].path == "src/test.py"