"""
Merge coordination module for DATS.

Provides conflict detection, classification, and resolution for
parallel task execution with optimistic concurrency.
"""

from src.merge.models import (
    ConflictRecord,
    ConflictClassification,
    ConflictResolution,
    ConflictType,
    ConflictSeverity,
    ResolutionStrategy,
    ResolutionStatus,
    AffectedArtifact,
    InvolvedTask,
)
from src.merge.config import MergeConfig, get_merge_config
from src.merge.detector import ConflictDetector
from src.merge.classifier import ConflictClassifier
from src.merge.coordinator import MergeCoordinator
from src.merge.strategies import ResolutionStrategySelector

__all__ = [
    # Models
    "ConflictRecord",
    "ConflictClassification",
    "ConflictResolution",
    "ConflictType",
    "ConflictSeverity",
    "ResolutionStrategy",
    "ResolutionStatus",
    "AffectedArtifact",
    "InvolvedTask",
    # Config
    "MergeConfig",
    "get_merge_config",
    # Core components
    "ConflictDetector",
    "ConflictClassifier",
    "MergeCoordinator",
    "ResolutionStrategySelector",
]