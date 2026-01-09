"""
Configuration for merge coordination.

Provides settings for conflict detection, resolution strategies,
escalation behavior, and batching.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.merge.models import ConflictSeverity


@dataclass
class DetectionConfig:
    """Configuration for conflict detection."""

    # When to check for conflicts
    check_on_task_complete: bool = True
    check_in_flight_overlap: bool = True

    # Semantic similarity threshold for detecting conceptual overlap
    semantic_similarity_threshold: float = 0.85

    # Minimum overlap for file-level detection (0.0-1.0)
    file_overlap_threshold: float = 0.1


@dataclass
class ResolutionConfig:
    """Configuration for conflict resolution."""

    # Maximum severity that can be auto-merged
    auto_merge_max_severity: ConflictSeverity = ConflictSeverity.TRIVIAL

    # Maximum severity for semantic merge
    semantic_merge_max_severity: ConflictSeverity = ConflictSeverity.MODERATE

    # Prefer human decision for architectural conflicts
    prefer_human_for_architectural: bool = True

    # Verify merged output (parse check for code)
    verify_merged_output: bool = True


@dataclass
class EscalationConfig:
    """Configuration for escalation behavior."""

    # Try frontier model before escalating to human
    escalate_to_frontier_before_human: bool = True

    # Maximum resolution attempts before escalation
    max_resolution_attempts: int = 2

    # Timeout for human review (hours)
    human_review_timeout_hours: int = 24


@dataclass
class BatchingConfig:
    """Configuration for conflict batching."""

    # Window for batching nearby task completions (seconds)
    # Can be overridden per-project
    default_batch_window_seconds: float = 30.0

    # Maximum conflicts in a batch
    max_batch_size: int = 10

    # Per-project overrides (project_id -> window_seconds)
    project_overrides: dict[str, float] = field(default_factory=dict)

    def get_window_for_project(self, project_id: str) -> float:
        """Get batch window for a specific project."""
        return self.project_overrides.get(
            project_id, self.default_batch_window_seconds
        )

    def set_window_for_project(self, project_id: str, window_seconds: float):
        """Set batch window for a specific project."""
        self.project_overrides[project_id] = window_seconds


@dataclass
class MergeConfig:
    """
    Complete configuration for merge coordination.

    Combines detection, resolution, escalation, and batching settings.
    """

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    # Storage path for conflict records
    storage_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detection": {
                "check_on_task_complete": self.detection.check_on_task_complete,
                "check_in_flight_overlap": self.detection.check_in_flight_overlap,
                "semantic_similarity_threshold": self.detection.semantic_similarity_threshold,
                "file_overlap_threshold": self.detection.file_overlap_threshold,
            },
            "resolution": {
                "auto_merge_max_severity": self.resolution.auto_merge_max_severity.value,
                "semantic_merge_max_severity": self.resolution.semantic_merge_max_severity.value,
                "prefer_human_for_architectural": self.resolution.prefer_human_for_architectural,
                "verify_merged_output": self.resolution.verify_merged_output,
            },
            "escalation": {
                "escalate_to_frontier_before_human": self.escalation.escalate_to_frontier_before_human,
                "max_resolution_attempts": self.escalation.max_resolution_attempts,
                "human_review_timeout_hours": self.escalation.human_review_timeout_hours,
            },
            "batching": {
                "default_batch_window_seconds": self.batching.default_batch_window_seconds,
                "max_batch_size": self.batching.max_batch_size,
                "project_overrides": self.batching.project_overrides,
            },
            "storage_path": self.storage_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeConfig":
        """Create from dictionary."""
        detection_data = data.get("detection", {})
        resolution_data = data.get("resolution", {})
        escalation_data = data.get("escalation", {})
        batching_data = data.get("batching", {})

        return cls(
            detection=DetectionConfig(
                check_on_task_complete=detection_data.get("check_on_task_complete", True),
                check_in_flight_overlap=detection_data.get("check_in_flight_overlap", True),
                semantic_similarity_threshold=detection_data.get(
                    "semantic_similarity_threshold", 0.85
                ),
                file_overlap_threshold=detection_data.get("file_overlap_threshold", 0.1),
            ),
            resolution=ResolutionConfig(
                auto_merge_max_severity=ConflictSeverity(
                    resolution_data.get("auto_merge_max_severity", "trivial")
                ),
                semantic_merge_max_severity=ConflictSeverity(
                    resolution_data.get("semantic_merge_max_severity", "moderate")
                ),
                prefer_human_for_architectural=resolution_data.get(
                    "prefer_human_for_architectural", True
                ),
                verify_merged_output=resolution_data.get("verify_merged_output", True),
            ),
            escalation=EscalationConfig(
                escalate_to_frontier_before_human=escalation_data.get(
                    "escalate_to_frontier_before_human", True
                ),
                max_resolution_attempts=escalation_data.get("max_resolution_attempts", 2),
                human_review_timeout_hours=escalation_data.get(
                    "human_review_timeout_hours", 24
                ),
            ),
            batching=BatchingConfig(
                default_batch_window_seconds=batching_data.get(
                    "default_batch_window_seconds", 30.0
                ),
                max_batch_size=batching_data.get("max_batch_size", 10),
                project_overrides=batching_data.get("project_overrides", {}),
            ),
            storage_path=data.get("storage_path"),
        )


# Global configuration instance
_merge_config: Optional[MergeConfig] = None


def get_merge_config() -> MergeConfig:
    """
    Get the global merge configuration.

    Creates a default configuration if none exists.

    Returns:
        MergeConfig instance
    """
    global _merge_config
    if _merge_config is None:
        _merge_config = MergeConfig()
    return _merge_config


def set_merge_config(config: MergeConfig):
    """
    Set the global merge configuration.

    Args:
        config: MergeConfig to use globally
    """
    global _merge_config
    _merge_config = config


def configure_merge(
    storage_path: Optional[str] = None,
    batch_window_seconds: float = 30.0,
    semantic_threshold: float = 0.85,
    max_resolution_attempts: int = 2,
    **kwargs,
) -> MergeConfig:
    """
    Convenience function to configure merge settings.

    Args:
        storage_path: Path for conflict record storage
        batch_window_seconds: Default batch window
        semantic_threshold: Semantic similarity threshold
        max_resolution_attempts: Max attempts before escalation
        **kwargs: Additional configuration overrides

    Returns:
        Configured MergeConfig instance
    """
    config = MergeConfig(
        detection=DetectionConfig(
            semantic_similarity_threshold=semantic_threshold,
        ),
        escalation=EscalationConfig(
            max_resolution_attempts=max_resolution_attempts,
        ),
        batching=BatchingConfig(
            default_batch_window_seconds=batch_window_seconds,
        ),
        storage_path=storage_path,
    )

    set_merge_config(config)
    return config