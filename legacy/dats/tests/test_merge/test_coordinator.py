"""Tests for merge coordinator."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.merge.config import MergeConfig
from src.merge.coordinator import MergeCoordinator, ResolutionOutcome
from src.merge.detector import TaskOutput
from src.merge.models import (
    AffectedArtifact,
    ConflictClassification,
    ConflictRecord,
    ConflictRegion,
    ConflictResolution,
    ConflictSeverity,
    ConflictType,
    InvolvedTask,
    ResolutionStatus,
    ResolutionStrategy,
)
from src.merge.resolvers.base import ResolverResult
from src.storage.work_product import Artifact


class TestMergeCoordinator:
    """Tests for MergeCoordinator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MergeConfig()

    @pytest.fixture
    def coordinator(self, config):
        """Create test coordinator."""
        return MergeCoordinator(config=config)

    @pytest.fixture
    def mock_conflict(self):
        """Create a mock conflict for testing."""
        return ConflictRecord(
            id="conflict-1",
            project_id="proj-1",
            involved_tasks=[
                InvolvedTask(
                    task_id="task-1",
                    output_artifact_id="art-1",
                    completed_at=datetime.utcnow(),
                    description="Implement feature A",
                ),
                InvolvedTask(
                    task_id="task-2",
                    output_artifact_id="art-2",
                    completed_at=datetime.utcnow(),
                    description="Implement feature B",
                ),
            ],
            affected_artifacts=[
                AffectedArtifact(
                    artifact_id="art-1",
                    location="src/module.py",
                    artifact_type="code",
                    conflict_regions=[
                        ConflictRegion(
                            start_line=10,
                            end_line=20,
                            ours_content="def func_a():\n    return 1\n",
                            theirs_content="def func_a():\n    return 2\n",
                        )
                    ],
                )
            ],
        )

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes components."""
        assert coordinator.detector is not None
        assert coordinator.classifier is not None
        assert coordinator.strategy_selector is not None
        assert "textual" in coordinator._resolvers
        assert "semantic" in coordinator._resolvers
        assert "architectural" in coordinator._resolvers

    def test_get_resolver_auto_merge(self, coordinator):
        """Test getting resolver for auto-merge strategy."""
        resolver = coordinator.get_resolver(ResolutionStrategy.AUTO_MERGE)
        assert resolver.resolver_name == "textual"

    def test_get_resolver_semantic(self, coordinator):
        """Test getting resolver for semantic merge strategy."""
        resolver = coordinator.get_resolver(ResolutionStrategy.SEMANTIC_MERGE)
        assert resolver.resolver_name == "semantic"

    def test_get_resolver_redesign(self, coordinator):
        """Test getting resolver for redesign strategy."""
        resolver = coordinator.get_resolver(ResolutionStrategy.REDESIGN)
        assert resolver.resolver_name == "architectural"

    def test_get_resolver_human_decision(self, coordinator):
        """Test getting resolver for human decision strategy."""
        resolver = coordinator.get_resolver(ResolutionStrategy.HUMAN_DECISION)
        assert resolver.resolver_name == "architectural"

    def test_conflict_storage(self, coordinator, mock_conflict):
        """Test conflict storage and retrieval."""
        coordinator._conflicts[mock_conflict.id] = mock_conflict

        retrieved = coordinator.get_conflict(mock_conflict.id)
        assert retrieved is not None
        assert retrieved.id == mock_conflict.id

        missing = coordinator.get_conflict("nonexistent")
        assert missing is None

    def test_get_pending_conflicts(self, coordinator, mock_conflict):
        """Test getting pending conflicts."""
        # Store conflict without resolution
        coordinator._conflicts[mock_conflict.id] = mock_conflict

        pending = coordinator.get_pending_conflicts()
        assert len(pending) == 1
        assert pending[0].id == mock_conflict.id

        # Filter by project
        pending_proj = coordinator.get_pending_conflicts(project_id="proj-1")
        assert len(pending_proj) == 1

        pending_other = coordinator.get_pending_conflicts(project_id="proj-2")
        assert len(pending_other) == 0

    def test_get_escalated_conflicts(self, coordinator, mock_conflict):
        """Test getting escalated conflicts."""
        # Add escalated resolution
        mock_conflict.resolution = ConflictResolution(
            strategy=ResolutionStrategy.HUMAN_DECISION,
            status=ResolutionStatus.ESCALATED,
        )
        coordinator._conflicts[mock_conflict.id] = mock_conflict

        escalated = coordinator.get_escalated_conflicts()
        assert len(escalated) == 1

        pending = coordinator.get_pending_conflicts()
        assert len(pending) == 0


class TestResolutionOutcome:
    """Tests for ResolutionOutcome dataclass."""

    def test_successful_outcome(self):
        """Test creating successful outcome."""
        outcome = ResolutionOutcome(
            conflict_id="conflict-1",
            success=True,
            attempts=1,
            total_tokens=100,
            total_duration_ms=500,
        )

        assert outcome.success
        assert outcome.conflict_id == "conflict-1"
        assert not outcome.escalated
        assert outcome.error is None

    def test_failed_outcome(self):
        """Test creating failed outcome."""
        outcome = ResolutionOutcome(
            conflict_id="conflict-1",
            success=False,
            error="Merge failed",
            attempts=2,
        )

        assert not outcome.success
        assert outcome.error == "Merge failed"

    def test_escalated_outcome(self):
        """Test creating escalated outcome."""
        outcome = ResolutionOutcome(
            conflict_id="conflict-1",
            success=True,
            escalated=True,
            escalation_target="human",
        )

        assert outcome.success
        assert outcome.escalated
        assert outcome.escalation_target == "human"


class TestClassificationIntegration:
    """Tests for classification in coordinator."""

    @pytest.fixture
    def mock_conflict(self):
        """Create conflict with classification."""
        conflict = ConflictRecord(
            id="conflict-1",
            project_id="proj-1",
            involved_tasks=[
                InvolvedTask(task_id="task-1", output_artifact_id="art-1"),
                InvolvedTask(task_id="task-2", output_artifact_id="art-2"),
            ],
            affected_artifacts=[
                AffectedArtifact(
                    artifact_id="art-1",
                    location="src/module.py",
                    conflict_regions=[
                        ConflictRegion(
                            start_line=1,
                            end_line=5,
                            ours_content="# Version A",
                            theirs_content="# Version B",
                        )
                    ],
                )
            ],
        )
        # Pre-classify as trivial textual
        conflict.classification = ConflictClassification(
            type=ConflictType.TEXTUAL,
            severity=ConflictSeverity.TRIVIAL,
            auto_resolvable=True,
            confidence=0.9,
        )
        return conflict

    def test_strategy_selection_trivial(self, mock_conflict):
        """Test strategy selection for trivial conflict."""
        from src.merge.strategies import ResolutionStrategySelector

        selector = ResolutionStrategySelector()
        recommendation = selector.select_strategy(
            conflict=mock_conflict,
            classification=mock_conflict.classification,
        )

        assert recommendation.strategy == ResolutionStrategy.AUTO_MERGE
        assert recommendation.confidence > 0.5


class TestMergeEngine:
    """Tests for DifflibMergeEngine."""

    @pytest.fixture
    def engine(self):
        """Create merge engine."""
        from src.merge.resolvers.textual import DifflibMergeEngine

        return DifflibMergeEngine()

    def test_two_way_merge_no_conflict(self, engine):
        """Test two-way merge with no overlapping changes."""
        ours = "line1\nline2\nline3\n"
        theirs = "line1\nline2\nline3\n"

        result = engine.two_way_merge(ours, theirs)

        assert result.success
        assert not result.has_conflicts
        assert result.content == ours

    def test_two_way_merge_with_conflict(self, engine):
        """Test two-way merge with conflict."""
        ours = "line1\nversion A\nline3\n"
        theirs = "line1\nversion B\nline3\n"

        result = engine.two_way_merge(ours, theirs)

        assert not result.success
        assert result.has_conflicts
        assert "<<<<<<< OURS" in result.content
        assert "=======" in result.content
        assert ">>>>>>> THEIRS" in result.content

    def test_compute_diff(self, engine):
        """Test diff computation."""
        old = "line1\nold line\nline3\n"
        new = "line1\nnew line\nline3\n"

        diff = engine.compute_diff(old, new)

        assert "-old line" in diff
        assert "+new line" in diff


class TestTextualResolver:
    """Tests for TextualResolver."""

    @pytest.fixture
    def resolver(self):
        """Create textual resolver."""
        from src.merge.resolvers.textual import TextualResolver

        return TextualResolver()

    @pytest.fixture
    def textual_conflict(self):
        """Create conflict for textual resolution."""
        conflict = ConflictRecord(
            id="conflict-1",
            project_id="proj-1",
            involved_tasks=[
                InvolvedTask(task_id="task-1", output_artifact_id="art-1"),
                InvolvedTask(task_id="task-2", output_artifact_id="art-2"),
            ],
            affected_artifacts=[
                AffectedArtifact(
                    artifact_id="art-1",
                    location="src/module.py",
                    artifact_type="code",
                    conflict_regions=[
                        ConflictRegion(
                            start_line=1,
                            end_line=3,
                            ours_content="def func():\n    return 1\n",
                            theirs_content="def func():\n    return 1\n",  # Same
                        )
                    ],
                )
            ],
        )
        conflict.classification = ConflictClassification(
            type=ConflictType.TEXTUAL,
            severity=ConflictSeverity.TRIVIAL,
            auto_resolvable=True,
            confidence=0.9,
        )
        return conflict

    def test_can_resolve_textual(self, resolver, textual_conflict):
        """Test can_resolve for textual conflict."""
        assert resolver.can_resolve(textual_conflict)

    def test_can_resolve_no_classification(self, resolver):
        """Test can_resolve without classification."""
        conflict = ConflictRecord()
        assert not resolver.can_resolve(conflict)


class TestStrategySelection:
    """Tests for strategy selection."""

    @pytest.fixture
    def selector(self):
        """Create strategy selector."""
        from src.merge.strategies import ResolutionStrategySelector

        return ResolutionStrategySelector()

    def test_trivial_textual_auto_merge(self, selector):
        """Trivial textual should recommend auto-merge."""
        conflict = ConflictRecord()
        classification = ConflictClassification(
            type=ConflictType.TEXTUAL,
            severity=ConflictSeverity.TRIVIAL,
            auto_resolvable=True,
            confidence=0.9,
        )

        recommendation = selector.select_strategy(conflict, classification)

        assert recommendation.strategy == ResolutionStrategy.AUTO_MERGE

    def test_moderate_semantic_merge(self, selector):
        """Moderate semantic should recommend semantic merge."""
        conflict = ConflictRecord()
        classification = ConflictClassification(
            type=ConflictType.SEMANTIC,
            severity=ConflictSeverity.MODERATE,
            auto_resolvable=False,
            confidence=0.8,
        )

        recommendation = selector.select_strategy(conflict, classification)

        assert recommendation.strategy == ResolutionStrategy.SEMANTIC_MERGE

    def test_fundamental_architectural_redesign(self, selector):
        """Fundamental architectural should recommend redesign or human."""
        conflict = ConflictRecord()
        classification = ConflictClassification(
            type=ConflictType.ARCHITECTURAL,
            severity=ConflictSeverity.FUNDAMENTAL,
            auto_resolvable=False,
            confidence=0.85,
        )

        recommendation = selector.select_strategy(conflict, classification)

        # Should escalate to human for architectural
        assert recommendation.strategy == ResolutionStrategy.HUMAN_DECISION

    def test_low_confidence_escalation(self, selector):
        """Low confidence should escalate."""
        conflict = ConflictRecord()
        classification = ConflictClassification(
            type=ConflictType.TEXTUAL,
            severity=ConflictSeverity.MODERATE,
            auto_resolvable=False,
            confidence=0.3,  # Low confidence
        )

        recommendation = selector.select_strategy(conflict, classification)

        assert recommendation.should_escalate


class TestConflictModels:
    """Tests for conflict data models."""

    def test_conflict_record_serialization(self):
        """Test ConflictRecord to_dict and from_dict."""
        conflict = ConflictRecord(
            id="conflict-1",
            project_id="proj-1",
            involved_tasks=[
                InvolvedTask(
                    task_id="task-1",
                    output_artifact_id="art-1",
                )
            ],
        )

        data = conflict.to_dict()
        restored = ConflictRecord.from_dict(data)

        assert restored.id == conflict.id
        assert restored.project_id == conflict.project_id
        assert len(restored.involved_tasks) == 1

    def test_conflict_classification_serialization(self):
        """Test ConflictClassification serialization."""
        classification = ConflictClassification(
            type=ConflictType.SEMANTIC,
            severity=ConflictSeverity.MODERATE,
            auto_resolvable=False,
            confidence=0.75,
            reasoning="Test reasoning",
        )

        data = classification.to_dict()
        restored = ConflictClassification.from_dict(data)

        assert restored.type == classification.type
        assert restored.severity == classification.severity
        assert restored.confidence == classification.confidence

    def test_conflict_is_resolved(self):
        """Test is_resolved check."""
        conflict = ConflictRecord()
        assert not conflict.is_resolved()

        conflict.resolution = ConflictResolution(
            strategy=ResolutionStrategy.AUTO_MERGE,
            status=ResolutionStatus.PENDING,
        )
        assert not conflict.is_resolved()

        conflict.resolution.status = ResolutionStatus.RESOLVED
        assert conflict.is_resolved()

    def test_conflict_is_escalated(self):
        """Test is_escalated check."""
        conflict = ConflictRecord()
        assert not conflict.is_escalated()

        conflict.resolution = ConflictResolution(
            strategy=ResolutionStrategy.HUMAN_DECISION,
            status=ResolutionStatus.ESCALATED,
        )
        assert conflict.is_escalated()