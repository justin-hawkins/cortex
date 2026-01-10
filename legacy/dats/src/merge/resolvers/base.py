"""
Base resolver for merge coordination.

Provides abstract interface for all conflict resolvers and
the MergeEngine protocol for swappable merge implementations.
"""

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    AffectedArtifact,
    ConflictRecord,
    ConflictResolution,
    MergedOutput,
    ResolutionStatus,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result from a merge operation."""

    success: bool
    content: str = ""
    has_conflicts: bool = False
    conflict_markers: list[tuple[int, int]] = field(default_factory=list)  # (start, end) lines
    error: Optional[str] = None


class MergeEngine(Protocol):
    """
    Protocol for merge engine implementations.

    This abstraction allows swapping between difflib (current) and
    git merge-file (future) implementations.
    """

    def three_way_merge(
        self,
        base: str,
        ours: str,
        theirs: str,
    ) -> MergeResult:
        """
        Perform a three-way merge.

        Args:
            base: Common ancestor content
            ours: Our version content
            theirs: Their version content

        Returns:
            MergeResult with merged content or conflict info
        """
        ...

    def compute_diff(
        self,
        old: str,
        new: str,
    ) -> str:
        """
        Compute diff between two versions.

        Args:
            old: Original content
            new: Modified content

        Returns:
            Diff string (unified format)
        """
        ...


@dataclass
class ResolverResult:
    """Result from a resolver operation."""

    success: bool
    resolution: Optional[ConflictResolution] = None
    error: Optional[str] = None
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None
    tokens_consumed: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "error": self.error,
            "needs_escalation": self.needs_escalation,
            "escalation_reason": self.escalation_reason,
            "tokens_consumed": self.tokens_consumed,
            "duration_ms": self.duration_ms,
        }


class BaseResolver(ABC):
    """
    Abstract base class for conflict resolvers.

    Each resolver implements resolution logic for specific
    conflict types and strategies.
    """

    resolver_name: str = "base"

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
    ):
        """
        Initialize resolver.

        Args:
            config: Merge configuration
        """
        self.config = config or get_merge_config()

    @abstractmethod
    async def resolve(
        self,
        conflict: ConflictRecord,
    ) -> ResolverResult:
        """
        Resolve a conflict.

        Args:
            conflict: Conflict to resolve

        Returns:
            ResolverResult with resolution details
        """
        pass

    @abstractmethod
    def can_resolve(
        self,
        conflict: ConflictRecord,
    ) -> bool:
        """
        Check if this resolver can handle the conflict.

        Args:
            conflict: Conflict to check

        Returns:
            True if this resolver can handle it
        """
        pass

    def verify_resolution(
        self,
        conflict: ConflictRecord,
        resolution: ConflictResolution,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify that a resolution is valid.

        Default implementation checks that merged output is syntactically valid
        for code files.

        Args:
            conflict: Original conflict
            resolution: Proposed resolution

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.resolution.verify_merged_output:
            return True, None

        if not resolution.merged_output:
            return True, None  # Nothing to verify

        # Check affected artifacts for language
        for artifact in conflict.affected_artifacts:
            if artifact.artifact_type != "code":
                continue

            content = resolution.merged_output.content

            # Try to parse as Python
            if self._looks_like_python(artifact.location):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    return False, f"Python syntax error: {e}"

            # Add more language checks as needed
            # For now, just check Python

        return True, None

    def _looks_like_python(self, path: str) -> bool:
        """Check if file path suggests Python code."""
        return path.endswith(".py")

    def _create_resolution(
        self,
        strategy: ResolutionStrategy,
        status: ResolutionStatus,
        merged_output: Optional[MergedOutput] = None,
        resolved_by: Optional[str] = None,
    ) -> ConflictResolution:
        """
        Create a ConflictResolution object.

        Args:
            strategy: Strategy used
            status: Resolution status
            merged_output: Optional merged output
            resolved_by: Who/what resolved it

        Returns:
            ConflictResolution instance
        """
        return ConflictResolution(
            strategy=strategy,
            status=status,
            merged_output=merged_output,
            resolved_by=resolved_by,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow() if status == ResolutionStatus.RESOLVED else None,
        )

    def _get_artifact_content(
        self,
        conflict: ConflictRecord,
        artifact_id: str,
    ) -> Optional[str]:
        """
        Get content for an artifact from the conflict.

        This is a placeholder - in production, would fetch from
        work product store.

        Args:
            conflict: Conflict record
            artifact_id: Artifact to get

        Returns:
            Artifact content if found
        """
        # For now, try to get from conflict regions
        for artifact in conflict.affected_artifacts:
            if artifact.artifact_id == artifact_id:
                # Return combined conflict content
                parts = []
                for region in artifact.conflict_regions:
                    if region.ours_content:
                        parts.append(region.ours_content)
                return "\n".join(parts) if parts else None
        return None


class VerificationError(Exception):
    """Raised when resolution verification fails."""

    pass