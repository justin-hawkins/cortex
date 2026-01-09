"""
Textual resolver for merge coordination.

Handles textual conflicts using standard merge algorithms
with difflib (swappable to git in the future).
"""

import difflib
import logging
from datetime import datetime
from typing import Optional

from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    ConflictRecord,
    ConflictType,
    MergedOutput,
    ResolutionStatus,
    ResolutionStrategy,
)
from src.merge.resolvers.base import (
    BaseResolver,
    MergeEngine,
    MergeResult,
    ResolverResult,
)

logger = logging.getLogger(__name__)


class DifflibMergeEngine:
    """
    Merge engine implementation using Python's difflib.

    Provides three-way merge functionality that can be swapped
    for git merge-file in the future.
    """

    def three_way_merge(
        self,
        base: str,
        ours: str,
        theirs: str,
    ) -> MergeResult:
        """
        Perform a three-way merge using difflib.

        This is a simplified implementation. For full three-way merge
        semantics, consider using the 'merge3' library or git merge-file.

        Args:
            base: Common ancestor content
            ours: Our version content
            theirs: Their version content

        Returns:
            MergeResult with merged content or conflict info
        """
        base_lines = base.splitlines(keepends=True)
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        # Use SequenceMatcher to find differences
        matcher_ours = difflib.SequenceMatcher(None, base_lines, ours_lines)
        matcher_theirs = difflib.SequenceMatcher(None, base_lines, theirs_lines)

        # Get matching blocks
        ours_changes = self._get_changes(matcher_ours, base_lines, ours_lines)
        theirs_changes = self._get_changes(matcher_theirs, base_lines, theirs_lines)

        # Try to merge
        merged_lines = []
        conflicts = []
        has_conflicts = False

        # Simple merge: if changes don't overlap, we can merge
        # Otherwise, mark as conflict
        i = 0
        while i < len(base_lines) or ours_changes or theirs_changes:
            # Check for changes at current position
            ours_change = self._find_change_at(ours_changes, i)
            theirs_change = self._find_change_at(theirs_changes, i)

            if ours_change and theirs_change:
                # Both modified - check if same change
                if ours_change["new"] == theirs_change["new"]:
                    # Same change - no conflict
                    merged_lines.extend(ours_change["new"])
                    i = ours_change["end"]
                    ours_changes.remove(ours_change)
                    theirs_changes.remove(theirs_change)
                else:
                    # Different changes - conflict!
                    has_conflicts = True
                    conflict_start = len(merged_lines)
                    
                    # Add conflict markers
                    merged_lines.append("<<<<<<< OURS\n")
                    merged_lines.extend(ours_change["new"])
                    merged_lines.append("=======\n")
                    merged_lines.extend(theirs_change["new"])
                    merged_lines.append(">>>>>>> THEIRS\n")
                    
                    conflict_end = len(merged_lines)
                    conflicts.append((conflict_start, conflict_end))
                    
                    i = max(ours_change["end"], theirs_change["end"])
                    ours_changes.remove(ours_change)
                    theirs_changes.remove(theirs_change)
            elif ours_change:
                # Only ours changed
                merged_lines.extend(ours_change["new"])
                i = ours_change["end"]
                ours_changes.remove(ours_change)
            elif theirs_change:
                # Only theirs changed
                merged_lines.extend(theirs_change["new"])
                i = theirs_change["end"]
                theirs_changes.remove(theirs_change)
            else:
                # No changes - use base
                if i < len(base_lines):
                    merged_lines.append(base_lines[i])
                i += 1

        return MergeResult(
            success=not has_conflicts,
            content="".join(merged_lines),
            has_conflicts=has_conflicts,
            conflict_markers=conflicts,
        )

    def _get_changes(
        self,
        matcher: difflib.SequenceMatcher,
        base_lines: list[str],
        new_lines: list[str],
    ) -> list[dict]:
        """Extract changes from a sequence matcher."""
        changes = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "insert", "delete"):
                changes.append({
                    "start": i1,
                    "end": i2,
                    "new": new_lines[j1:j2],
                    "tag": tag,
                })
        return changes

    def _find_change_at(
        self,
        changes: list[dict],
        pos: int,
    ) -> Optional[dict]:
        """Find a change that starts at or covers position."""
        for change in changes:
            if change["start"] <= pos < change["end"]:
                return change
            if change["start"] == pos and change["end"] == pos:
                # Insert at this position
                return change
        return None

    def compute_diff(
        self,
        old: str,
        new: str,
    ) -> str:
        """
        Compute unified diff between two versions.

        Args:
            old: Original content
            new: Modified content

        Returns:
            Unified diff string
        """
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
        )
        return "".join(diff)

    def two_way_merge(
        self,
        ours: str,
        theirs: str,
    ) -> MergeResult:
        """
        Perform a two-way merge (no base).

        Attempts to combine non-overlapping changes.

        Args:
            ours: Our version content
            theirs: Their version content

        Returns:
            MergeResult with merged content or conflict info
        """
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, ours_lines, theirs_lines)
        merged_lines = []
        has_conflicts = False
        conflicts = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                merged_lines.extend(ours_lines[i1:i2])
            elif tag == "replace":
                # Conflict
                has_conflicts = True
                conflict_start = len(merged_lines)
                merged_lines.append("<<<<<<< OURS\n")
                merged_lines.extend(ours_lines[i1:i2])
                merged_lines.append("=======\n")
                merged_lines.extend(theirs_lines[j1:j2])
                merged_lines.append(">>>>>>> THEIRS\n")
                conflict_end = len(merged_lines)
                conflicts.append((conflict_start, conflict_end))
            elif tag == "delete":
                # Ours has content, theirs doesn't
                merged_lines.extend(ours_lines[i1:i2])
            elif tag == "insert":
                # Theirs has content, ours doesn't
                merged_lines.extend(theirs_lines[j1:j2])

        return MergeResult(
            success=not has_conflicts,
            content="".join(merged_lines),
            has_conflicts=has_conflicts,
            conflict_markers=conflicts,
        )


class TextualResolver(BaseResolver):
    """
    Resolves textual conflicts using standard merge algorithms.

    Used for:
    - Non-overlapping file changes (auto-merge)
    - Simple overlapping changes where three-way merge succeeds
    """

    resolver_name = "textual"

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        merge_engine: Optional[MergeEngine] = None,
    ):
        """
        Initialize textual resolver.

        Args:
            config: Merge configuration
            merge_engine: Optional custom merge engine
        """
        super().__init__(config)
        self.merge_engine = merge_engine or DifflibMergeEngine()

    def can_resolve(
        self,
        conflict: ConflictRecord,
    ) -> bool:
        """
        Check if this resolver can handle the conflict.

        Textual resolver handles:
        - Textual conflicts
        - Trivial severity conflicts
        - Conflicts with auto_resolvable=True

        Args:
            conflict: Conflict to check

        Returns:
            True if can handle
        """
        if not conflict.classification:
            return False

        classification = conflict.classification

        # Can handle textual conflicts
        if classification.type == ConflictType.TEXTUAL:
            return True

        # Can handle if marked auto-resolvable
        if classification.auto_resolvable:
            return True

        return False

    async def resolve(
        self,
        conflict: ConflictRecord,
    ) -> ResolverResult:
        """
        Resolve a textual conflict.

        Args:
            conflict: Conflict to resolve

        Returns:
            ResolverResult with resolution details
        """
        started_at = datetime.utcnow()

        try:
            # Process each affected artifact
            merged_contents = []
            all_success = True
            merge_notes = []

            for artifact in conflict.affected_artifacts:
                if not artifact.conflict_regions:
                    # No conflict regions - nothing to merge
                    continue

                # Get content from conflict regions
                # In production, we'd fetch full file content from work product store
                ours_content = self._get_ours_content(artifact)
                theirs_content = self._get_theirs_content(artifact)
                base_content = self._get_base_content(artifact)

                # Attempt merge
                if base_content:
                    result = self.merge_engine.three_way_merge(
                        base=base_content,
                        ours=ours_content,
                        theirs=theirs_content,
                    )
                else:
                    # No base - try two-way merge
                    result = self.merge_engine.two_way_merge(
                        ours=ours_content,
                        theirs=theirs_content,
                    )

                if result.success:
                    merged_contents.append({
                        "path": artifact.location,
                        "content": result.content,
                    })
                    merge_notes.append(
                        f"Successfully merged {artifact.location}"
                    )
                else:
                    all_success = False
                    merge_notes.append(
                        f"Conflicts remain in {artifact.location}: "
                        f"{len(result.conflict_markers)} conflict(s)"
                    )
                    # Include partial result with markers
                    merged_contents.append({
                        "path": artifact.location,
                        "content": result.content,
                        "has_conflicts": True,
                    })

            duration = datetime.utcnow() - started_at

            if all_success and merged_contents:
                # Create successful resolution
                merged_output = MergedOutput(
                    artifact_type="code",
                    content=merged_contents[0]["content"] if len(merged_contents) == 1 else "",
                    merge_notes="\n".join(merge_notes),
                    artifacts_produced=[m["path"] for m in merged_contents],
                )

                resolution = self._create_resolution(
                    strategy=ResolutionStrategy.AUTO_MERGE,
                    status=ResolutionStatus.RESOLVED,
                    merged_output=merged_output,
                    resolved_by="textual_resolver",
                )

                # Verify resolution
                is_valid, error = self.verify_resolution(conflict, resolution)
                if not is_valid:
                    return ResolverResult(
                        success=False,
                        error=f"Verification failed: {error}",
                        needs_escalation=True,
                        escalation_reason="Auto-merge produced invalid output",
                        duration_ms=int(duration.total_seconds() * 1000),
                    )

                return ResolverResult(
                    success=True,
                    resolution=resolution,
                    duration_ms=int(duration.total_seconds() * 1000),
                )
            else:
                # Merge had conflicts - needs escalation
                return ResolverResult(
                    success=False,
                    error="Auto-merge produced conflicts",
                    needs_escalation=True,
                    escalation_reason="\n".join(merge_notes),
                    duration_ms=int(duration.total_seconds() * 1000),
                )

        except Exception as e:
            logger.error(f"Textual resolution failed: {e}")
            duration = datetime.utcnow() - started_at
            return ResolverResult(
                success=False,
                error=str(e),
                needs_escalation=True,
                escalation_reason=f"Textual resolver exception: {e}",
                duration_ms=int(duration.total_seconds() * 1000),
            )

    def _get_ours_content(self, artifact) -> str:
        """Get 'ours' content from artifact."""
        parts = []
        for region in artifact.conflict_regions:
            if region.ours_content:
                parts.append(region.ours_content)
        return "\n".join(parts) if parts else ""

    def _get_theirs_content(self, artifact) -> str:
        """Get 'theirs' content from artifact."""
        parts = []
        for region in artifact.conflict_regions:
            if region.theirs_content:
                parts.append(region.theirs_content)
        return "\n".join(parts) if parts else ""

    def _get_base_content(self, artifact) -> str:
        """Get base content from artifact."""
        parts = []
        for region in artifact.conflict_regions:
            if region.base_content:
                parts.append(region.base_content)
        return "\n".join(parts) if parts else ""