"""
Tests for provenance tracking and cascade failure handling.
"""

import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from src.storage.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceGraph,
    ArtifactRef,
    InputRef,
    TaintEvent,
    Checkpoint,
    VerificationStatus,
    ArtifactType,
    InputRelationship,
    compute_checksum,
)
from src.cascade.detector import (
    CascadeDetector,
    CascadeScenario,
    CascadeMetrics,
    CascadeTrigger,
    CascadeSeverity,
)
from src.cascade.taint import TaintPropagator, PropagationResult
from src.cascade.revalidation import (
    RevalidationTask,
    RevalidationQueue,
    RevalidationVerdict,
)
from src.cascade.rollback import RollbackManager, RollbackTrigger, RollbackResult


class TestProvenanceRecord:
    """Tests for ProvenanceRecord dataclass."""

    def test_create_empty_record(self):
        """Test creating a record with defaults."""
        record = ProvenanceRecord()
        assert record.id
        assert record.verification_status == "pending"
        assert not record.is_tainted()
        assert not record.is_suspect()
        assert record.is_clean()

    def test_create_record_with_values(self):
        """Test creating a record with explicit values."""
        record = ProvenanceRecord(
            task_id="task-123",
            project_id="project-456",
            model_used="openai/gpt-oss-20b",
            worker_id="code-general",
        )
        assert record.task_id == "task-123"
        assert record.project_id == "project-456"
        assert record.model_used == "openai/gpt-oss-20b"
        assert record.execution.model == "openai/gpt-oss-20b"

    def test_record_serialization(self):
        """Test to_dict/from_dict round-trip."""
        record = ProvenanceRecord(
            task_id="task-123",
            project_id="project-456",
            model_used="test-model",
            worker_id="test-worker",
        )
        record.outputs.append(ArtifactRef(
            artifact_id="artifact-1",
            type=ArtifactType.CODE,
            checksum="abc123",
        ))
        record.inputs.append(InputRef(
            artifact_id="input-1",
            relationship=InputRelationship.CONSUMED,
        ))

        data = record.to_dict()
        restored = ProvenanceRecord.from_dict(data)

        assert restored.task_id == record.task_id
        assert restored.project_id == record.project_id
        assert len(restored.outputs) == 1
        assert restored.outputs[0].artifact_id == "artifact-1"
        assert len(restored.inputs) == 1

    def test_taint_status(self):
        """Test taint status methods."""
        record = ProvenanceRecord()
        assert record.is_clean()
        
        record.taint.is_tainted = True
        assert record.is_tainted()
        assert not record.is_clean()
        
        record.taint.is_tainted = False
        record.taint.is_suspect = True
        assert record.is_suspect()
        assert not record.is_clean()


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a tracker with temp storage."""
        return ProvenanceTracker(storage_path=str(tmp_path / "provenance"))

    def test_create_record(self, tracker):
        """Test creating a new provenance record."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        
        assert record.id
        assert record.task_id == "task-1"
        assert record.started_at is not None
        
        # Should be retrievable
        retrieved = tracker.get_record(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id

    def test_complete_record(self, tracker):
        """Test completing a provenance record."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"artifact_id": "artifact-1", "type": "code"}],
            tokens_input=100,
            tokens_output=200,
            confidence=0.9,
        )
        
        updated = tracker.get_record(record.id)
        assert updated.completed_at is not None
        assert len(updated.outputs) == 1
        assert updated.tokens_input == 100
        assert updated.tokens_output == 200

    def test_add_inputs(self, tracker):
        """Test adding inputs to a record."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        
        tracker.add_inputs(
            record_id=record.id,
            input_artifact_ids=["input-1", "input-2"],
            checksums={"input-1": "abc123"},
        )
        
        updated = tracker.get_record(record.id)
        assert len(updated.inputs) == 2
        assert updated.inputs[0].artifact_id == "input-1"
        assert updated.inputs[0].version_at_consumption == "abc123"

    def test_mark_tainted(self, tracker):
        """Test marking a record as tainted."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"artifact_id": "artifact-1"}],
        )
        
        tracker.mark_tainted(
            record_id=record.id,
            reason="Test taint",
            source_id="source-1",
        )
        
        updated = tracker.get_record(record.id)
        assert updated.is_tainted()
        assert updated.taint.tainted_reason == "Test taint"
        assert updated.verification_status == "tainted"

    def test_mark_suspect(self, tracker):
        """Test marking a record as suspect."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"artifact_id": "artifact-1"}],
        )
        
        tracker.mark_suspect(
            record_id=record.id,
            source_artifact_id="tainted-artifact",
        )
        
        updated = tracker.get_record(record.id)
        assert updated.is_suspect()
        assert updated.taint.suspect_source == "tainted-artifact"

    def test_clear_suspect(self, tracker):
        """Test clearing suspect status."""
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"artifact_id": "artifact-1"}],
        )
        
        tracker.mark_suspect(
            record_id=record.id,
            source_artifact_id="tainted-artifact",
        )
        
        tracker.clear_suspect(
            record_id=record.id,
            reason="Revalidation passed",
        )
        
        updated = tracker.get_record(record.id)
        assert not updated.is_suspect()
        assert updated.is_clean()

    def test_get_by_project(self, tracker):
        """Test getting records by project."""
        for i in range(3):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
        
        # Another project
        tracker.create_record(
            task_id="task-other",
            project_id="project-2",
            model_used="test-model",
            worker_id="test-worker",
        )
        
        records = tracker.get_by_project("project-1")
        assert len(records) == 3

    def test_consistency_check(self, tracker):
        """Test project consistency check."""
        for i in range(5):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
            tracker.complete_record(
                record_id=record.id,
                outputs=[{"artifact_id": f"artifact-{i}"}],
            )
        
        # Taint one
        records = tracker.get_by_project("project-1")
        tracker.mark_tainted(records[0].id, "Test")
        
        report = tracker.consistency_check("project-1")
        assert report.total_records == 5
        assert report.tainted_count == 1
        assert report.recommended_action == "continue"


class TestProvenanceGraph:
    """Tests for ProvenanceGraph."""

    @pytest.fixture
    def tracker_with_graph(self, tmp_path):
        """Create a tracker with linked provenance records."""
        tracker = ProvenanceTracker(storage_path=str(tmp_path / "provenance"))
        
        # Create a chain: artifact-1 -> artifact-2 -> artifact-3
        record1 = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(
            record_id=record1.id,
            outputs=[{"artifact_id": "artifact-1"}],
        )
        
        record2 = tracker.create_record(
            task_id="task-2",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.add_inputs(record2.id, ["artifact-1"])
        tracker.complete_record(
            record_id=record2.id,
            outputs=[{"artifact_id": "artifact-2"}],
        )
        
        record3 = tracker.create_record(
            task_id="task-3",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.add_inputs(record3.id, ["artifact-2"])
        tracker.complete_record(
            record_id=record3.id,
            outputs=[{"artifact_id": "artifact-3"}],
        )
        
        return tracker

    def test_find_dependents(self, tracker_with_graph):
        """Test finding dependent artifacts."""
        graph = tracker_with_graph.graph
        
        dependents = graph.find_dependents("artifact-1", transitive=True)
        assert "artifact-2" in dependents
        assert "artifact-3" in dependents
        
        direct_only = graph.find_dependents("artifact-1", transitive=False)
        assert "artifact-2" in direct_only
        assert "artifact-3" not in direct_only

    def test_find_dependencies(self, tracker_with_graph):
        """Test finding dependency artifacts."""
        graph = tracker_with_graph.graph
        
        dependencies = graph.find_dependencies("artifact-3", transitive=True)
        assert "artifact-2" in dependencies
        assert "artifact-1" in dependencies

    def test_find_path(self, tracker_with_graph):
        """Test finding path between artifacts."""
        graph = tracker_with_graph.graph
        
        path = graph.find_path("artifact-1", "artifact-3")
        assert path == ["artifact-1", "artifact-2", "artifact-3"]
        
        no_path = graph.find_path("artifact-3", "artifact-1")
        assert no_path == []


class TestCascadeDetector:
    """Tests for CascadeDetector."""

    @pytest.fixture
    def detector_with_data(self, tmp_path):
        """Create a detector with test data."""
        tracker = ProvenanceTracker(storage_path=str(tmp_path / "provenance"))
        
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(
            record_id=record.id,
            outputs=[{"artifact_id": "artifact-1"}],
        )
        tracker.update_verification(record.id, "passed")
        
        detector = CascadeDetector(tracker)
        return detector, tracker

    def test_detect_from_manual_taint(self, detector_with_data):
        """Test detecting cascade from manual taint."""
        detector, tracker = detector_with_data
        
        scenario = detector.detect_from_manual_taint(
            artifact_id="artifact-1",
            reason="Testing taint",
            requested_by="tester",
        )
        
        assert scenario is not None
        assert scenario.source_artifact_id == "artifact-1"
        assert scenario.trigger == CascadeTrigger.MANUAL_TAINT
        assert scenario.reason == "Testing taint"

    def test_estimate_impact(self, detector_with_data):
        """Test impact estimation."""
        detector, tracker = detector_with_data
        
        metrics = detector.estimate_impact("artifact-1")
        
        assert isinstance(metrics, CascadeMetrics)
        assert not metrics.exceeds_thresholds()

    def test_severity_determination(self, detector_with_data):
        """Test severity is determined correctly."""
        detector, _ = detector_with_data
        
        assert detector._determine_severity(0, 0) == CascadeSeverity.LOW
        assert detector._determine_severity(10, 2) == CascadeSeverity.MEDIUM
        assert detector._determine_severity(30, 4) == CascadeSeverity.HIGH
        assert detector._determine_severity(100, 10) == CascadeSeverity.CRITICAL


class TestTaintPropagator:
    """Tests for TaintPropagator."""

    @pytest.fixture
    def propagator_with_chain(self, tmp_path):
        """Create a propagator with a chain of artifacts."""
        tracker = ProvenanceTracker(storage_path=str(tmp_path / "provenance"))
        
        # Create chain
        for i in range(3):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
            if i > 0:
                tracker.add_inputs(record.id, [f"artifact-{i-1}"])
            tracker.complete_record(
                record_id=record.id,
                outputs=[{"artifact_id": f"artifact-{i}"}],
            )
            tracker.update_verification(record.id, "passed")
        
        propagator = TaintPropagator(tracker)
        return propagator, tracker

    def test_taint_artifact(self, propagator_with_chain):
        """Test tainting an artifact and propagating."""
        propagator, tracker = propagator_with_chain
        
        result = propagator.taint_artifact(
            artifact_id="artifact-0",
            reason="Test taint",
        )
        
        assert result.tainted_count == 1
        assert "artifact-0" in result.tainted_artifact_ids
        assert result.suspect_count == 2  # artifact-1 and artifact-2
        assert "artifact-1" in result.suspect_artifact_ids
        assert "artifact-2" in result.suspect_artifact_ids

    def test_clear_suspect_status(self, propagator_with_chain):
        """Test clearing suspect status."""
        propagator, tracker = propagator_with_chain
        
        # First taint
        propagator.taint_artifact("artifact-0", "Test")
        
        # Clear one suspect
        cleared = propagator.clear_suspect_status("artifact-1", "Validated OK")
        
        assert cleared
        record = tracker.get_producer("artifact-1")
        assert not record.is_suspect()


class TestRevalidationQueue:
    """Tests for RevalidationQueue."""

    @pytest.fixture
    def queue(self, tmp_path):
        """Create a queue with temp storage."""
        return RevalidationQueue(
            storage_path=str(tmp_path / "queue"),
            max_depth=5,
            max_count=50,
        )

    def test_add_and_pop(self, queue):
        """Test adding and popping tasks."""
        task = RevalidationTask(
            suspect_artifact_id="artifact-1",
            taint_source_artifact_id="source-1",
            cascade_depth=1,
        )
        
        task_id = queue.add(task)
        assert task_id
        assert queue.get_pending_count() == 1
        
        popped = queue.pop()
        assert popped.suspect_artifact_id == "artifact-1"
        assert popped.status == "processing"
        assert queue.get_pending_count() == 0

    def test_priority_ordering(self, queue):
        """Test that tasks are ordered by priority."""
        # Depth-first ordering means higher depth = higher priority
        task1 = RevalidationTask(
            suspect_artifact_id="artifact-1",
            cascade_depth=1,
        )
        task2 = RevalidationTask(
            suspect_artifact_id="artifact-2",
            cascade_depth=3,
        )
        task3 = RevalidationTask(
            suspect_artifact_id="artifact-3",
            cascade_depth=2,
        )
        
        queue.add(task1)
        queue.add(task2)
        queue.add(task3)
        
        # Should get depth 3 first
        first = queue.pop()
        assert first.cascade_depth == 3

    def test_complete_task(self, queue):
        """Test completing a task."""
        task = RevalidationTask(suspect_artifact_id="artifact-1")
        queue.add(task)
        
        popped = queue.pop()
        queue.complete(popped, RevalidationVerdict.STILL_VALID, "OK")
        
        completed = queue.get_task(popped.id)
        assert completed.status == "completed"
        assert completed.verdict == RevalidationVerdict.STILL_VALID

    def test_exceeds_thresholds(self, queue):
        """Test threshold detection."""
        assert not queue.exceeds_thresholds()
        
        # Add many tasks
        for i in range(60):
            task = RevalidationTask(
                suspect_artifact_id=f"artifact-{i}",
                cascade_depth=i % 3,
            )
            queue.add(task)
        
        assert queue.exceeds_thresholds()  # Exceeds max_count of 50


class TestRollbackManager:
    """Tests for RollbackManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a manager with test data."""
        tracker = ProvenanceTracker(storage_path=str(tmp_path / "provenance"))
        manager = RollbackManager(
            provenance_tracker=tracker,
            storage_path=str(tmp_path / "rollback"),
            auto_checkpoint=False,
        )
        return manager, tracker

    def test_create_checkpoint(self, manager):
        """Test checkpoint creation."""
        rollback_manager, tracker = manager
        
        # Create some records
        for i in range(5):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
            tracker.complete_record(
                record_id=record.id,
                outputs=[{"artifact_id": f"artifact-{i}"}],
            )
        
        checkpoint = rollback_manager.create_checkpoint(
            project_id="project-1",
            description="Test checkpoint",
            git_tag=False,
        )
        
        assert checkpoint.id
        assert checkpoint.project_id == "project-1"
        assert checkpoint.provenance_count == 5

    def test_rollback_dry_run(self, manager):
        """Test rollback impact estimation."""
        rollback_manager, tracker = manager
        
        # Create checkpoint
        for i in range(3):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
            tracker.complete_record(
                record_id=record.id,
                outputs=[{"artifact_id": f"artifact-{i}"}],
            )
        
        checkpoint = rollback_manager.create_checkpoint(
            project_id="project-1",
            git_tag=False,
        )
        
        # Create more records after checkpoint
        for i in range(3, 6):
            record = tracker.create_record(
                task_id=f"task-{i}",
                project_id="project-1",
                model_used="test-model",
                worker_id="test-worker",
            )
            tracker.complete_record(
                record_id=record.id,
                outputs=[{"artifact_id": f"artifact-{i}"}],
            )
        
        # Dry run should show what would be rolled back
        result = rollback_manager.rollback_to_checkpoint(
            checkpoint_id=checkpoint.id,
            dry_run=True,
        )
        
        assert result.success
        assert result.provenance_records_invalidated == 3

    def test_get_best_rollback_point(self, manager):
        """Test finding best rollback point."""
        rollback_manager, tracker = manager
        
        # Create checkpoints
        record = tracker.create_record(
            task_id="task-1",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(record.id, outputs=[])
        
        checkpoint1 = rollback_manager.create_checkpoint("project-1", git_tag=False)
        
        record2 = tracker.create_record(
            task_id="task-2",
            project_id="project-1",
            model_used="test-model",
            worker_id="test-worker",
        )
        tracker.complete_record(record2.id, outputs=[])
        
        checkpoint2 = rollback_manager.create_checkpoint("project-1", git_tag=False)
        
        best = rollback_manager.get_best_rollback_point("project-1")
        assert best.id == checkpoint2.id  # Most recent


class TestChecksum:
    """Tests for checksum utility."""

    def test_compute_checksum_string(self):
        """Test checksum of string content."""
        checksum = compute_checksum("test content")
        assert len(checksum) == 64  # SHA256 hex digest
        
        # Same content should give same checksum
        assert compute_checksum("test content") == checksum

    def test_compute_checksum_bytes(self):
        """Test checksum of bytes content."""
        checksum = compute_checksum(b"test content")
        assert len(checksum) == 64