"""
Work product storage for DATS.

Provides Git/storage abstraction for artifacts produced by workers.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class Artifact:
    """Represents a work product artifact."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "code"  # code, document, config, analysis, architecture
    content: str = ""
    path: Optional[str] = None
    language: Optional[str] = None
    checksum: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Compute checksum if not provided."""
        if self.content and not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "path": self.path,
            "language": self.language,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", "code"),
            content=data.get("content", ""),
            path=data.get("path"),
            language=data.get("language"),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )


class WorkProductStore:
    """
    Store and manage work product artifacts.

    Provides file-based storage with optional Git integration.
    """

    def __init__(
        self,
        base_path: str,
        use_git: bool = False,
        github_token: Optional[str] = None,
    ):
        """
        Initialize work product store.

        Args:
            base_path: Base directory for artifact storage
            use_git: Enable Git integration
            github_token: Optional GitHub token for remote operations
        """
        self.base_path = Path(base_path)
        self.use_git = use_git
        self.github_token = github_token
        self._artifacts: dict[str, Artifact] = {}

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create index file if it doesn't exist
        self._index_path = self.base_path / ".artifact_index.json"
        if not self._index_path.exists():
            self._save_index()

    def store(
        self,
        content: str,
        artifact_type: str = "code",
        path: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """
        Store a new artifact.

        Args:
            content: Artifact content
            artifact_type: Type of artifact
            path: Optional relative path for the artifact
            language: Optional programming language
            metadata: Optional metadata

        Returns:
            Created Artifact
        """
        artifact = Artifact(
            type=artifact_type,
            content=content,
            path=path,
            language=language,
            metadata=metadata or {},
        )

        # Determine storage path
        if path:
            file_path = self.base_path / path
        else:
            # Generate path from ID
            ext = self._get_extension(artifact_type, language)
            file_path = self.base_path / "artifacts" / f"{artifact.id}{ext}"

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        file_path.write_text(content)
        artifact.path = str(file_path.relative_to(self.base_path))

        # Store in memory
        self._artifacts[artifact.id] = artifact

        # Update index
        self._save_index()

        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """
        Get an artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact if found
        """
        if artifact_id in self._artifacts:
            return self._artifacts[artifact_id]

        # Try loading from index
        self._load_index()
        return self._artifacts.get(artifact_id)

    def get_by_path(self, path: str) -> Optional[Artifact]:
        """
        Get an artifact by path.

        Args:
            path: Relative path

        Returns:
            Artifact if found
        """
        for artifact in self._artifacts.values():
            if artifact.path == path:
                return artifact
        return None

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[Artifact]:
        """
        List artifacts with optional filters.

        Args:
            artifact_type: Optional type filter
            language: Optional language filter

        Returns:
            List of matching artifacts
        """
        self._load_index()
        
        artifacts = list(self._artifacts.values())
        
        if artifact_type:
            artifacts = [a for a in artifacts if a.type == artifact_type]
        
        if language:
            artifacts = [a for a in artifacts if a.language == language]
        
        return artifacts

    def update(
        self,
        artifact_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """
        Update an existing artifact.

        Args:
            artifact_id: Artifact to update
            content: New content
            metadata: Optional metadata updates

        Returns:
            Updated Artifact
        """
        artifact = self.get(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")

        # Update content
        artifact.content = content
        artifact.checksum = hashlib.sha256(content.encode()).hexdigest()
        
        if metadata:
            artifact.metadata.update(metadata)

        # Write to file
        if artifact.path:
            file_path = self.base_path / artifact.path
            file_path.write_text(content)

        # Update index
        self._save_index()

        return artifact

    def delete(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Artifact to delete

        Returns:
            True if deleted, False if not found
        """
        artifact = self._artifacts.pop(artifact_id, None)
        if not artifact:
            return False

        # Delete file
        if artifact.path:
            file_path = self.base_path / artifact.path
            if file_path.exists():
                file_path.unlink()

        # Update index
        self._save_index()

        return True

    def _save_index(self):
        """Save artifact index to disk."""
        index = {
            "artifacts": {
                aid: art.to_dict()
                for aid, art in self._artifacts.items()
            },
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_index(self):
        """Load artifact index from disk."""
        if not self._index_path.exists():
            return

        with open(self._index_path) as f:
            index = json.load(f)

        for aid, data in index.get("artifacts", {}).items():
            if aid not in self._artifacts:
                self._artifacts[aid] = Artifact.from_dict(data)

    @staticmethod
    def _get_extension(artifact_type: str, language: Optional[str]) -> str:
        """Get file extension for artifact."""
        if language:
            extensions = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "rust": ".rs",
                "go": ".go",
                "c": ".c",
                "cpp": ".cpp",
                "java": ".java",
                "markdown": ".md",
                "json": ".json",
                "yaml": ".yaml",
                "html": ".html",
                "css": ".css",
            }
            return extensions.get(language.lower(), ".txt")

        type_extensions = {
            "code": ".txt",
            "document": ".md",
            "config": ".yaml",
            "analysis": ".json",
            "architecture": ".md",
        }
        return type_extensions.get(artifact_type, ".txt")