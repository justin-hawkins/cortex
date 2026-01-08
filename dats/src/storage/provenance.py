"""
Provenance tracking for DATS.

Tracks lineage of all outputs for traceability and rollback.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProvenanceRecord:
    """Record of task execution provenance."""

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    project_id: str = ""

    # What was produced
    outputs: list[dict[str, Any]] = field(default_factory=list)

    # Lineage
    inputs_consumed: list[str] = field(default_factory=list)

    # Execution details
    model_used: str = ""
    worker_id: str = ""
    prompt_template_name: Optional[str] = None
    prompt_template_version: Optional[str] = None

    # Quality indicators
    confidence: float = 0.0
    verification_status: str = "pending"
    verification_details: Optional[dict[str, Any]] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    tokens_input: int = 0
    tokens_output: int = 0
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "project_id": self.project_id,
            "outputs": self.outputs,
            "inputs_consumed": self.inputs_consumed,
            "model_used": self.model_used,
            "worker_id": self.worker_id,
            "prompt_template": {
                "name": self.prompt_template_name,
                "version": self.prompt_template_version,
            },
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "verification_details": self.verification_details,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary."""
        prompt = data.get("prompt_template", {})
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            project_id=data.get("project_id", ""),
            outputs=data.get("outputs", []),
            inputs_consumed=data.get("inputs_consumed", []),
            model_used=data.get("model_used", ""),
            worker_id=data.get("worker_id", ""),
            prompt_template_name=prompt.get("name"),
            prompt_template_version=prompt.get("version"),
            confidence=data.get("confidence", 0.0),
            verification_status=data.get("verification_status", "pending"),
            verification_details=data.get("verification_details"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            execution_time_ms=data.get("execution_time_ms"),
        )


class ProvenanceTracker:
    """
    Track and store provenance records.

    Provides methods for creating, storing, and querying provenance.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize provenance tracker.

        Args:
            storage_path: Optional path for file-based storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: dict[str, ProvenanceRecord] = {}

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_record(
        self,
        task_id: str,
        project_id: str,
        model_used: str,
        worker_id: str,
    ) -> ProvenanceRecord:
        """
        Create a new provenance record.

        Args:
            task_id: Associated task ID
            project_id: Associated project ID
            model_used: Model that was used
            worker_id: Worker that executed the task

        Returns:
            New ProvenanceRecord
        """
        record = ProvenanceRecord(
            task_id=task_id,
            project_id=project_id,
            model_used=model_used,
            worker_id=worker_id,
            started_at=datetime.utcnow(),
        )
        self._records[record.id] = record
        return record

    def complete_record(
        self,
        record_id: str,
        outputs: list[dict[str, Any]],
        tokens_input: int = 0,
        tokens_output: int = 0,
        confidence: float = 0.0,
    ) -> ProvenanceRecord:
        """
        Complete a provenance record with results.

        Args:
            record_id: ID of record to complete
            outputs: List of output artifacts
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            confidence: Confidence score

        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        record.completed_at = datetime.utcnow()
        record.outputs = outputs
        record.tokens_input = tokens_input
        record.tokens_output = tokens_output
        record.confidence = confidence

        if record.started_at and record.completed_at:
            delta = record.completed_at - record.started_at
            record.execution_time_ms = int(delta.total_seconds() * 1000)

        # Persist if storage configured
        if self.storage_path:
            self._save_record(record)

        return record

    def update_verification(
        self,
        record_id: str,
        status: str,
        details: Optional[dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Update verification status of a record.

        Args:
            record_id: ID of record to update
            status: New verification status
            details: Optional verification details

        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        record.verification_status = status
        record.verification_details = details

        if self.storage_path:
            self._save_record(record)

        return record

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """
        Get a provenance record by ID.

        Args:
            record_id: Record ID

        Returns:
            ProvenanceRecord if found
        """
        # Check memory first
        if record_id in self._records:
            return self._records[record_id]

        # Try loading from disk
        if self.storage_path:
            return self._load_record(record_id)

        return None

    def get_by_task(self, task_id: str) -> list[ProvenanceRecord]:
        """
        Get all provenance records for a task.

        Args:
            task_id: Task ID

        Returns:
            List of ProvenanceRecords
        """
        records = []
        for record in self._records.values():
            if record.task_id == task_id:
                records.append(record)
        return records

    def get_lineage(self, artifact_id: str) -> list[ProvenanceRecord]:
        """
        Get lineage (ancestry) of an artifact.

        Args:
            artifact_id: ID of artifact to trace

        Returns:
            List of ProvenanceRecords in lineage order
        """
        lineage = []
        visited = set()

        def trace(aid: str):
            if aid in visited:
                return
            visited.add(aid)

            # Find record that produced this artifact
            for record in self._records.values():
                for output in record.outputs:
                    if output.get("artifact_id") == aid:
                        lineage.append(record)
                        # Trace inputs
                        for input_id in record.inputs_consumed:
                            trace(input_id)
                        break

        trace(artifact_id)
        return lineage

    def _save_record(self, record: ProvenanceRecord):
        """Save record to disk."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"{record.id}.json"
        with open(file_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _load_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Load record from disk."""
        if not self.storage_path:
            return None

        file_path = self.storage_path / f"{record_id}.json"
        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)
            record = ProvenanceRecord.from_dict(data)
            self._records[record.id] = record
            return record


def compute_checksum(content: str | bytes) -> str:
    """
    Compute SHA256 checksum for content.

    Args:
        content: String or bytes to hash

    Returns:
        Hex digest of SHA256 hash
    """
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()