"""
Provenance tracking for DATS.

Tracks lineage of all outputs for traceability, taint propagation, and rollback.
Provides DAG operations for impact analysis and root cause investigation.
"""

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class VerificationStatus(str, Enum):
    """Verification status for provenance records."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    HUMAN_APPROVED = "human_approved"
    TAINTED = "tainted"
    SUSPECT = "suspect"  # Potentially affected by upstream taint


class InputRelationship(str, Enum):
    """Type of relationship between input and output."""

    CONSUMED = "consumed"  # Directly used in production
    REFERENCED = "referenced"  # Referenced but not directly consumed


class ArtifactType(str, Enum):
    """Type of artifact produced."""

    CODE = "code"
    DOCUMENT = "document"
    CONFIG = "config"
    ANALYSIS = "analysis"
    ARCHITECTURE = "architecture"


@dataclass
class ArtifactRef:
    """Reference to an artifact with version tracking."""

    artifact_id: str
    type: ArtifactType = ArtifactType.CODE
    location: str = ""  # Path in work product store
    checksum: str = ""  # SHA256 for integrity
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "type": self.type.value if isinstance(self.type, ArtifactType) else self.type,
            "location": self.location,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        """Create from dictionary."""
        artifact_type = data.get("type", "code")
        if isinstance(artifact_type, str):
            try:
                artifact_type = ArtifactType(artifact_type)
            except ValueError:
                artifact_type = ArtifactType.CODE
        return cls(
            artifact_id=data.get("artifact_id", ""),
            type=artifact_type,
            location=data.get("location", ""),
            checksum=data.get("checksum", ""),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class InputRef:
    """Reference to an input artifact with lineage tracking."""

    artifact_id: str
    relationship: InputRelationship = InputRelationship.CONSUMED
    version_at_consumption: str = ""  # Checksum at time of use

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "relationship": self.relationship.value if isinstance(self.relationship, InputRelationship) else self.relationship,
            "version_at_consumption": self.version_at_consumption,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputRef":
        """Create from dictionary."""
        relationship = data.get("relationship", "consumed")
        if isinstance(relationship, str):
            try:
                relationship = InputRelationship(relationship)
            except ValueError:
                relationship = InputRelationship.CONSUMED
        return cls(
            artifact_id=data.get("artifact_id", ""),
            relationship=relationship,
            version_at_consumption=data.get("version_at_consumption", ""),
        )


@dataclass
class ExecutionContext:
    """Context about task execution."""

    model: str = ""  # e.g., "openai/gpt-oss-20b"
    worker_id: str = ""
    prompt_template_name: str = ""
    prompt_template_version: str = ""
    prompt_template_hash: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "worker_id": self.worker_id,
            "prompt_template": {
                "name": self.prompt_template_name,
                "version": self.prompt_template_version,
                "hash": self.prompt_template_hash,
            },
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        """Create from dictionary."""
        prompt = data.get("prompt_template", {})
        return cls(
            model=data.get("model", ""),
            worker_id=data.get("worker_id", ""),
            prompt_template_name=prompt.get("name", ""),
            prompt_template_version=prompt.get("version", ""),
            prompt_template_hash=prompt.get("hash", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class VerificationInfo:
    """Information about quality verification."""

    status: VerificationStatus = VerificationStatus.PENDING
    qa_profile: str = ""
    qa_result_id: str = ""
    verified_at: Optional[datetime] = None
    verified_by: str = ""  # Model or human ID
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value if isinstance(self.status, VerificationStatus) else self.status,
            "qa_profile": self.qa_profile,
            "qa_result_id": self.qa_result_id,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verified_by": self.verified_by,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationInfo":
        """Create from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            try:
                status = VerificationStatus(status)
            except ValueError:
                status = VerificationStatus.PENDING
        return cls(
            status=status,
            qa_profile=data.get("qa_profile", ""),
            qa_result_id=data.get("qa_result_id", ""),
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            verified_by=data.get("verified_by", ""),
            details=data.get("details", {}),
        )


@dataclass
class TaintInfo:
    """Information about taint status."""

    is_tainted: bool = False
    tainted_at: Optional[datetime] = None
    tainted_reason: str = ""
    tainted_by: str = ""  # Task ID or artifact ID that caused taint
    is_suspect: bool = False  # Potentially affected but not confirmed
    suspect_since: Optional[datetime] = None
    suspect_source: str = ""  # What tainted artifact made this suspect

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_tainted": self.is_tainted,
            "tainted_at": self.tainted_at.isoformat() if self.tainted_at else None,
            "tainted_reason": self.tainted_reason,
            "tainted_by": self.tainted_by,
            "is_suspect": self.is_suspect,
            "suspect_since": self.suspect_since.isoformat() if self.suspect_since else None,
            "suspect_source": self.suspect_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaintInfo":
        """Create from dictionary."""
        return cls(
            is_tainted=data.get("is_tainted", False),
            tainted_at=datetime.fromisoformat(data["tainted_at"]) if data.get("tainted_at") else None,
            tainted_reason=data.get("tainted_reason", ""),
            tainted_by=data.get("tainted_by", ""),
            is_suspect=data.get("is_suspect", False),
            suspect_since=datetime.fromisoformat(data["suspect_since"]) if data.get("suspect_since") else None,
            suspect_source=data.get("suspect_source", ""),
        )


@dataclass
class ExecutionMetrics:
    """Metrics about task execution."""

    tokens_input: int = 0
    tokens_output: int = 0
    execution_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionMetrics":
        """Create from dictionary."""
        return cls(
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            execution_time_ms=data.get("execution_time_ms", 0),
        )


@dataclass
class ProvenanceRecord:
    """
    Record of task execution provenance.
    
    Enhanced with taint tracking, structured inputs/outputs, and detailed
    execution context for full lineage traceability.
    """

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    project_id: str = ""

    # Outputs produced (enhanced: structured artifacts)
    outputs: list[ArtifactRef] = field(default_factory=list)
    
    # Legacy: simple list of output dicts for backward compatibility
    _legacy_outputs: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # Inputs consumed (enhanced: structured with versioning)
    inputs: list[InputRef] = field(default_factory=list)
    
    # Legacy: simple list of artifact IDs for backward compatibility
    inputs_consumed: list[str] = field(default_factory=list)

    # Execution context (enhanced)
    execution: ExecutionContext = field(default_factory=ExecutionContext)
    
    # Legacy fields for backward compatibility
    model_used: str = ""
    worker_id: str = ""
    prompt_template_name: Optional[str] = None
    prompt_template_version: Optional[str] = None

    # Verification (enhanced)
    verification: VerificationInfo = field(default_factory=VerificationInfo)
    
    # Legacy fields for backward compatibility
    confidence: float = 0.0
    verification_status: str = "pending"
    verification_details: Optional[dict[str, Any]] = None

    # Taint tracking (new)
    taint: TaintInfo = field(default_factory=TaintInfo)

    # Timestamps
    created_at: Optional[datetime] = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics (enhanced)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    
    # Legacy fields for backward compatibility
    tokens_input: int = 0
    tokens_output: int = 0
    execution_time_ms: Optional[int] = None

    def __post_init__(self):
        """Synchronize legacy and new fields."""
        # Sync execution context
        if self.model_used and not self.execution.model:
            self.execution.model = self.model_used
        elif self.execution.model and not self.model_used:
            self.model_used = self.execution.model
            
        if self.worker_id and not self.execution.worker_id:
            self.execution.worker_id = self.worker_id
        elif self.execution.worker_id and not self.worker_id:
            self.worker_id = self.execution.worker_id

        # Sync prompt template
        if self.prompt_template_name and not self.execution.prompt_template_name:
            self.execution.prompt_template_name = self.prompt_template_name
        if self.prompt_template_version and not self.execution.prompt_template_version:
            self.execution.prompt_template_version = self.prompt_template_version

        # Sync verification
        if self.verification_status != "pending" and self.verification.status == VerificationStatus.PENDING:
            try:
                self.verification.status = VerificationStatus(self.verification_status)
            except ValueError:
                pass
        elif self.verification.status != VerificationStatus.PENDING:
            self.verification_status = self.verification.status.value

        if self.confidence and not self.verification.details.get("confidence"):
            self.verification.details["confidence"] = self.confidence
            
        if self.verification_details and not self.verification.details:
            self.verification.details = self.verification_details

        # Sync metrics
        if self.tokens_input and not self.metrics.tokens_input:
            self.metrics.tokens_input = self.tokens_input
        elif self.metrics.tokens_input and not self.tokens_input:
            self.tokens_input = self.metrics.tokens_input
            
        if self.tokens_output and not self.metrics.tokens_output:
            self.metrics.tokens_output = self.tokens_output
        elif self.metrics.tokens_output and not self.tokens_output:
            self.tokens_output = self.metrics.tokens_output
            
        if self.execution_time_ms and not self.metrics.execution_time_ms:
            self.metrics.execution_time_ms = self.execution_time_ms
        elif self.metrics.execution_time_ms and not self.execution_time_ms:
            self.execution_time_ms = self.metrics.execution_time_ms

    def get_output_artifact_ids(self) -> list[str]:
        """Get all output artifact IDs."""
        ids = [o.artifact_id for o in self.outputs if o.artifact_id]
        # Also check legacy outputs
        for o in self._legacy_outputs:
            if o.get("artifact_id") and o["artifact_id"] not in ids:
                ids.append(o["artifact_id"])
        return ids

    def get_input_artifact_ids(self) -> list[str]:
        """Get all input artifact IDs."""
        ids = [i.artifact_id for i in self.inputs if i.artifact_id]
        # Also include legacy inputs_consumed
        for aid in self.inputs_consumed:
            if aid not in ids:
                ids.append(aid)
        return ids

    def is_tainted(self) -> bool:
        """Check if this record is tainted."""
        return self.taint.is_tainted

    def is_suspect(self) -> bool:
        """Check if this record is suspect (potentially tainted)."""
        return self.taint.is_suspect

    def is_clean(self) -> bool:
        """Check if this record is clean (not tainted or suspect)."""
        return not self.taint.is_tainted and not self.taint.is_suspect

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (backward compatible)."""
        result = {
            "id": self.id,
            "task_id": self.task_id,
            "project_id": self.project_id,
            # New structured outputs
            "outputs": [o.to_dict() for o in self.outputs] if self.outputs else self._legacy_outputs,
            # New structured inputs
            "inputs": [i.to_dict() for i in self.inputs],
            # Legacy inputs for backward compatibility
            "inputs_consumed": self.inputs_consumed or self.get_input_artifact_ids(),
            # Execution context
            "execution": self.execution.to_dict(),
            # Legacy execution fields
            "model_used": self.model_used or self.execution.model,
            "worker_id": self.worker_id or self.execution.worker_id,
            "prompt_template": {
                "name": self.prompt_template_name or self.execution.prompt_template_name,
                "version": self.prompt_template_version or self.execution.prompt_template_version,
            },
            # Verification
            "verification": self.verification.to_dict(),
            # Legacy verification fields
            "confidence": self.confidence or self.verification.details.get("confidence", 0.0),
            "verification_status": self.verification_status or self.verification.status.value,
            "verification_details": self.verification_details or self.verification.details,
            # Taint tracking
            "taint": self.taint.to_dict(),
            # Timestamps
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            # Metrics
            "metrics": self.metrics.to_dict(),
            # Legacy metrics fields
            "tokens_input": self.tokens_input or self.metrics.tokens_input,
            "tokens_output": self.tokens_output or self.metrics.tokens_output,
            "execution_time_ms": self.execution_time_ms or self.metrics.execution_time_ms,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary (backward compatible)."""
        # Parse outputs - handle both new and legacy format
        outputs = []
        legacy_outputs = []
        for o in data.get("outputs", []):
            if isinstance(o, dict):
                if "artifact_id" in o:
                    outputs.append(ArtifactRef.from_dict(o))
                else:
                    legacy_outputs.append(o)

        # Parse inputs - handle both new and legacy format
        inputs = []
        for i in data.get("inputs", []):
            if isinstance(i, dict):
                inputs.append(InputRef.from_dict(i))

        # Parse execution context
        execution = ExecutionContext.from_dict(data.get("execution", {}))
        
        # Parse verification
        verification = VerificationInfo.from_dict(data.get("verification", {}))
        
        # Parse taint info
        taint = TaintInfo.from_dict(data.get("taint", {}))
        
        # Parse metrics
        metrics = ExecutionMetrics.from_dict(data.get("metrics", {}))

        # Legacy prompt template
        prompt = data.get("prompt_template", {})

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            project_id=data.get("project_id", ""),
            outputs=outputs,
            _legacy_outputs=legacy_outputs,
            inputs=inputs,
            inputs_consumed=data.get("inputs_consumed", []),
            execution=execution,
            model_used=data.get("model_used", ""),
            worker_id=data.get("worker_id", ""),
            prompt_template_name=prompt.get("name"),
            prompt_template_version=prompt.get("version"),
            verification=verification,
            confidence=data.get("confidence", 0.0),
            verification_status=data.get("verification_status", "pending"),
            verification_details=data.get("verification_details"),
            taint=taint,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metrics=metrics,
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            execution_time_ms=data.get("execution_time_ms"),
        )


@dataclass
class TaintEvent:
    """
    Audit record for taint propagation.
    
    Immutable record of when and why artifacts were tainted.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # What was tainted
    artifact_id: str = ""
    provenance_id: str = ""
    
    # Why it was tainted
    reason: str = ""
    source_artifact_id: str = ""  # The tainted artifact that caused this
    source_provenance_id: str = ""
    
    # Cascade info
    cascade_depth: int = 0  # How far from original taint source
    cascade_id: str = ""  # Groups related taint events
    
    # Action taken
    action: str = "taint"  # taint, suspect, clear, escalate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "artifact_id": self.artifact_id,
            "provenance_id": self.provenance_id,
            "reason": self.reason,
            "source_artifact_id": self.source_artifact_id,
            "source_provenance_id": self.source_provenance_id,
            "cascade_depth": self.cascade_depth,
            "cascade_id": self.cascade_id,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaintEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            artifact_id=data.get("artifact_id", ""),
            provenance_id=data.get("provenance_id", ""),
            reason=data.get("reason", ""),
            source_artifact_id=data.get("source_artifact_id", ""),
            source_provenance_id=data.get("source_provenance_id", ""),
            cascade_depth=data.get("cascade_depth", 0),
            cascade_id=data.get("cascade_id", ""),
            action=data.get("action", "taint"),
        )


@dataclass
class Checkpoint:
    """
    Snapshot metadata for rollback points.
    
    Captures project state at a point in time for potential rollback.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # What's included
    git_ref: str = ""  # Git tag or commit hash
    provenance_count: int = 0  # Number of provenance records at checkpoint
    artifact_count: int = 0  # Number of artifacts at checkpoint
    
    # Snapshot data
    last_provenance_id: str = ""  # Last provenance record ID included
    last_task_id: str = ""  # Last task ID included
    
    # State references
    rag_state_ref: str = ""  # Reference to LightRAG state
    work_product_ref: str = ""  # Reference to work product store state
    
    # Metadata
    description: str = ""
    trigger: str = ""  # auto, manual, milestone
    is_valid: bool = True  # Can be invalidated if state is corrupted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "git_ref": self.git_ref,
            "provenance_count": self.provenance_count,
            "artifact_count": self.artifact_count,
            "last_provenance_id": self.last_provenance_id,
            "last_task_id": self.last_task_id,
            "rag_state_ref": self.rag_state_ref,
            "work_product_ref": self.work_product_ref,
            "description": self.description,
            "trigger": self.trigger,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            project_id=data.get("project_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            git_ref=data.get("git_ref", ""),
            provenance_count=data.get("provenance_count", 0),
            artifact_count=data.get("artifact_count", 0),
            last_provenance_id=data.get("last_provenance_id", ""),
            last_task_id=data.get("last_task_id", ""),
            rag_state_ref=data.get("rag_state_ref", ""),
            work_product_ref=data.get("work_product_ref", ""),
            description=data.get("description", ""),
            trigger=data.get("trigger", ""),
            is_valid=data.get("is_valid", True),
        )


@dataclass
class ImpactReport:
    """Report from impact analysis."""

    artifact_id: str = ""
    direct_dependents: list[str] = field(default_factory=list)
    transitive_dependents: list[str] = field(default_factory=list)
    total_affected: int = 0
    max_depth: int = 0
    tasks_in_flight: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "direct_dependents": self.direct_dependents,
            "transitive_dependents": self.transitive_dependents,
            "total_affected": self.total_affected,
            "max_depth": self.max_depth,
            "tasks_in_flight": self.tasks_in_flight,
        }


@dataclass
class RootCauseReport:
    """Report from root cause analysis."""

    failed_artifact_id: str = ""
    immediate_cause: str = ""
    root_cause: str = ""
    path: list[str] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failed_artifact_id": self.failed_artifact_id,
            "immediate_cause": self.immediate_cause,
            "root_cause": self.root_cause,
            "path": self.path,
            "depth": self.depth,
        }


@dataclass
class ConsistencyReport:
    """Report from project consistency check."""

    project_id: str = ""
    total_records: int = 0
    tainted_count: int = 0
    suspect_count: int = 0
    pending_revalidation: int = 0
    orphaned_artifacts: int = 0
    recommended_action: str = "continue"  # continue, pause, rollback

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "total_records": self.total_records,
            "tainted_count": self.tainted_count,
            "suspect_count": self.suspect_count,
            "pending_revalidation": self.pending_revalidation,
            "orphaned_artifacts": self.orphaned_artifacts,
            "recommended_action": self.recommended_action,
        }


class ProvenanceGraph:
    """
    DAG operations for provenance traversal.
    
    Provides efficient graph operations for impact analysis,
    root cause investigation, and cascade planning.
    """

    def __init__(self, tracker: "ProvenanceTracker"):
        """
        Initialize graph with tracker reference.
        
        Args:
            tracker: ProvenanceTracker to query for records
        """
        self.tracker = tracker
        self._forward_edges: dict[str, set[str]] = defaultdict(set)  # artifact -> dependents
        self._backward_edges: dict[str, set[str]] = defaultdict(set)  # artifact -> dependencies
        self._artifact_to_provenance: dict[str, str] = {}  # artifact_id -> provenance_id
        self._loaded = False

    def _load_graph(self):
        """Build graph from all provenance records."""
        if self._loaded:
            return

        for record in self.tracker.get_all_records():
            output_ids = record.get_output_artifact_ids()
            input_ids = record.get_input_artifact_ids()

            # Map artifacts to provenance
            for oid in output_ids:
                self._artifact_to_provenance[oid] = record.id

            # Build edges
            for oid in output_ids:
                for iid in input_ids:
                    self._forward_edges[iid].add(oid)  # input -> output (forward)
                    self._backward_edges[oid].add(iid)  # output -> input (backward)

        self._loaded = True

    def invalidate_cache(self):
        """Clear cached graph data."""
        self._forward_edges.clear()
        self._backward_edges.clear()
        self._artifact_to_provenance.clear()
        self._loaded = False

    def find_dependents(self, artifact_id: str, transitive: bool = True) -> list[str]:
        """
        Find all artifacts that depend on the given artifact.
        
        Args:
            artifact_id: Starting artifact
            transitive: If True, find all transitive dependents
            
        Returns:
            List of dependent artifact IDs
        """
        self._load_graph()
        
        if not transitive:
            return list(self._forward_edges.get(artifact_id, set()))

        visited = set()
        result = []
        queue = list(self._forward_edges.get(artifact_id, set()))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self._forward_edges.get(current, set()))

        return result

    def find_dependencies(self, artifact_id: str, transitive: bool = True) -> list[str]:
        """
        Find all artifacts that the given artifact depends on.
        
        Args:
            artifact_id: Starting artifact
            transitive: If True, find all transitive dependencies
            
        Returns:
            List of dependency artifact IDs
        """
        self._load_graph()
        
        if not transitive:
            return list(self._backward_edges.get(artifact_id, set()))

        visited = set()
        result = []
        queue = list(self._backward_edges.get(artifact_id, set()))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self._backward_edges.get(current, set()))

        return result

    def find_path(self, from_artifact: str, to_artifact: str) -> list[str]:
        """
        Find path between two artifacts in the DAG.
        
        Args:
            from_artifact: Starting artifact
            to_artifact: Target artifact
            
        Returns:
            List of artifact IDs forming path, empty if no path
        """
        self._load_graph()
        
        # BFS for shortest path
        visited = {from_artifact}
        queue = [(from_artifact, [from_artifact])]

        while queue:
            current, path = queue.pop(0)
            
            if current == to_artifact:
                return path

            for next_artifact in self._forward_edges.get(current, set()):
                if next_artifact not in visited:
                    visited.add(next_artifact)
                    queue.append((next_artifact, path + [next_artifact]))

        return []

    def get_task_tree(self, root_task_id: str) -> dict[str, Any]:
        """
        Get complete subgraph for a task tree.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Dictionary with nodes (provenance records) and edges
        """
        self._load_graph()
        
        nodes = []
        edges = []
        visited_records = set()

        # Get root record
        root_records = self.tracker.get_by_task(root_task_id)
        
        def traverse_from_record(record: ProvenanceRecord):
            if record.id in visited_records:
                return
            visited_records.add(record.id)
            nodes.append(record.to_dict())

            for output_id in record.get_output_artifact_ids():
                for dependent_id in self._forward_edges.get(output_id, set()):
                    prov_id = self._artifact_to_provenance.get(dependent_id)
                    if prov_id:
                        edges.append({
                            "from_artifact": output_id,
                            "to_artifact": dependent_id,
                            "from_provenance": record.id,
                            "to_provenance": prov_id,
                        })
                        dep_record = self.tracker.get_record(prov_id)
                        if dep_record:
                            traverse_from_record(dep_record)

        for record in root_records:
            traverse_from_record(record)

        return {
            "root_task_id": root_task_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def topological_sort(self, artifact_ids: Optional[list[str]] = None) -> list[str]:
        """
        Get artifacts in topological order (dependencies first).
        
        Args:
            artifact_ids: Optional subset to sort. If None, sorts all.
            
        Returns:
            List of artifact IDs in topological order
        """
        self._load_graph()
        
        if artifact_ids is None:
            artifact_ids = list(set(self._forward_edges.keys()) | set(self._backward_edges.keys()))

        artifact_set = set(artifact_ids)
        in_degree = {aid: 0 for aid in artifact_ids}
        
        # Calculate in-degrees within subset
        for aid in artifact_ids:
            for dep in self._backward_edges.get(aid, set()):
                if dep in artifact_set:
                    in_degree[aid] += 1

        # Kahn's algorithm
        queue = [aid for aid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for dependent in self._forward_edges.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return result

    def get_provenance_for_artifact(self, artifact_id: str) -> Optional[str]:
        """Get provenance record ID for an artifact."""
        self._load_graph()
        return self._artifact_to_provenance.get(artifact_id)


class ProvenanceTracker:
    """
    Track and store provenance records.

    Provides methods for creating, storing, and querying provenance.
    Enhanced with taint tracking, graph operations, and analysis queries.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize provenance tracker.

        Args:
            storage_path: Optional path for file-based storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: dict[str, ProvenanceRecord] = {}
        self._taint_events: list[TaintEvent] = []
        self._checkpoints: dict[str, Checkpoint] = {}
        self._graph: Optional[ProvenanceGraph] = None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._taint_events_path = self.storage_path / "taint_events.json"
            self._checkpoints_path = self.storage_path / "checkpoints.json"
            self._load_taint_events()
            self._load_checkpoints()

    @property
    def graph(self) -> ProvenanceGraph:
        """Get or create provenance graph."""
        if self._graph is None:
            self._graph = ProvenanceGraph(self)
        return self._graph

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
        
        # Invalidate graph cache
        if self._graph:
            self._graph.invalidate_cache()
            
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
        
        # Handle outputs - convert to ArtifactRef if possible
        for output in outputs:
            if isinstance(output, dict):
                if "artifact_id" in output:
                    record.outputs.append(ArtifactRef.from_dict(output))
                else:
                    record._legacy_outputs.append(output)
        
        record.tokens_input = tokens_input
        record.tokens_output = tokens_output
        record.metrics.tokens_input = tokens_input
        record.metrics.tokens_output = tokens_output
        record.confidence = confidence

        if record.started_at and record.completed_at:
            delta = record.completed_at - record.started_at
            record.execution_time_ms = int(delta.total_seconds() * 1000)
            record.metrics.execution_time_ms = record.execution_time_ms

        # Persist if storage configured
        if self.storage_path:
            self._save_record(record)

        # Invalidate graph cache
        if self._graph:
            self._graph.invalidate_cache()

        return record

    def add_inputs(
        self,
        record_id: str,
        input_artifact_ids: list[str],
        checksums: Optional[dict[str, str]] = None,
    ) -> ProvenanceRecord:
        """
        Add input artifacts to a record.
        
        Args:
            record_id: Record ID
            input_artifact_ids: List of input artifact IDs
            checksums: Optional mapping of artifact_id -> checksum
            
        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        checksums = checksums or {}
        
        for aid in input_artifact_ids:
            record.inputs.append(InputRef(
                artifact_id=aid,
                relationship=InputRelationship.CONSUMED,
                version_at_consumption=checksums.get(aid, ""),
            ))
            if aid not in record.inputs_consumed:
                record.inputs_consumed.append(aid)

        if self.storage_path:
            self._save_record(record)

        # Invalidate graph cache
        if self._graph:
            self._graph.invalidate_cache()

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
        
        # Also update new verification structure
        try:
            record.verification.status = VerificationStatus(status)
        except ValueError:
            pass
        record.verification.details = details or {}
        record.verification.verified_at = datetime.utcnow()

        if self.storage_path:
            self._save_record(record)

        return record

    def mark_tainted(
        self,
        record_id: str,
        reason: str,
        source_id: str = "",
        cascade_id: str = "",
        cascade_depth: int = 0,
    ) -> ProvenanceRecord:
        """
        Mark a provenance record as tainted.
        
        Args:
            record_id: Record to taint
            reason: Reason for tainting
            source_id: ID of artifact/record that caused the taint
            cascade_id: ID grouping related taint events
            cascade_depth: Depth in cascade from original taint
            
        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        record.taint.is_tainted = True
        record.taint.tainted_at = datetime.utcnow()
        record.taint.tainted_reason = reason
        record.taint.tainted_by = source_id
        record.taint.is_suspect = False  # Clear suspect if it was set
        
        record.verification.status = VerificationStatus.TAINTED
        record.verification_status = "tainted"

        # Log taint event
        for artifact_id in record.get_output_artifact_ids():
            event = TaintEvent(
                artifact_id=artifact_id,
                provenance_id=record_id,
                reason=reason,
                source_artifact_id=source_id,
                cascade_depth=cascade_depth,
                cascade_id=cascade_id or str(uuid.uuid4()),
                action="taint",
            )
            self._taint_events.append(event)

        if self.storage_path:
            self._save_record(record)
            self._save_taint_events()

        return record

    def mark_suspect(
        self,
        record_id: str,
        source_artifact_id: str,
        cascade_id: str = "",
        cascade_depth: int = 0,
    ) -> ProvenanceRecord:
        """
        Mark a provenance record as suspect (potentially affected by taint).
        
        Args:
            record_id: Record to mark
            source_artifact_id: Tainted artifact that might affect this
            cascade_id: ID grouping related taint events
            cascade_depth: Depth in cascade
            
        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        # Don't mark as suspect if already tainted
        if record.taint.is_tainted:
            return record

        record.taint.is_suspect = True
        record.taint.suspect_since = datetime.utcnow()
        record.taint.suspect_source = source_artifact_id
        
        record.verification.status = VerificationStatus.SUSPECT
        record.verification_status = "suspect"

        # Log taint event
        for artifact_id in record.get_output_artifact_ids():
            event = TaintEvent(
                artifact_id=artifact_id,
                provenance_id=record_id,
                reason=f"Dependent on tainted artifact {source_artifact_id}",
                source_artifact_id=source_artifact_id,
                cascade_depth=cascade_depth,
                cascade_id=cascade_id,
                action="suspect",
            )
            self._taint_events.append(event)

        if self.storage_path:
            self._save_record(record)
            self._save_taint_events()

        return record

    def clear_suspect(
        self,
        record_id: str,
        reason: str = "Revalidation passed",
    ) -> ProvenanceRecord:
        """
        Clear suspect status from a record after successful revalidation.
        
        Args:
            record_id: Record to clear
            reason: Reason for clearing
            
        Returns:
            Updated ProvenanceRecord
        """
        record = self._records.get(record_id)
        if not record:
            raise ValueError(f"Record not found: {record_id}")

        if not record.taint.is_suspect:
            return record

        source = record.taint.suspect_source
        record.taint.is_suspect = False
        record.taint.suspect_since = None
        record.taint.suspect_source = ""
        
        # Restore previous verification status
        record.verification.status = VerificationStatus.PASSED
        record.verification_status = "passed"

        # Log clear event
        for artifact_id in record.get_output_artifact_ids():
            event = TaintEvent(
                artifact_id=artifact_id,
                provenance_id=record_id,
                reason=reason,
                source_artifact_id=source,
                action="clear",
            )
            self._taint_events.append(event)

        if self.storage_path:
            self._save_record(record)
            self._save_taint_events()

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
        # Load all records first
        self._load_all_records()
        
        records = []
        for record in self._records.values():
            if record.task_id == task_id:
                records.append(record)
        return records

    def get_by_project(self, project_id: str) -> list[ProvenanceRecord]:
        """
        Get all provenance records for a project.

        Args:
            project_id: Project ID

        Returns:
            List of ProvenanceRecords
        """
        self._load_all_records()
        
        records = []
        for record in self._records.values():
            if record.project_id == project_id:
                records.append(record)
        return records

    def get_all_records(self) -> list[ProvenanceRecord]:
        """Get all provenance records."""
        self._load_all_records()
        return list(self._records.values())

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
                    if output.artifact_id == aid:
                        lineage.append(record)
                        # Trace inputs
                        for input_ref in record.inputs:
                            trace(input_ref.artifact_id)
                        for input_id in record.inputs_consumed:
                            trace(input_id)
                        break

        trace(artifact_id)
        return lineage

    # Query methods

    def get_producer(self, artifact_id: str) -> Optional[ProvenanceRecord]:
        """
        Get the provenance record that produced an artifact.
        
        Args:
            artifact_id: Artifact ID
            
        Returns:
            ProvenanceRecord if found
        """
        self._load_all_records()
        
        for record in self._records.values():
            if artifact_id in record.get_output_artifact_ids():
                return record
        return None

    def impact_analysis(self, artifact_id: str) -> ImpactReport:
        """
        Analyze impact if an artifact were tainted.
        
        Args:
            artifact_id: Artifact to analyze
            
        Returns:
            ImpactReport with affected artifacts
        """
        direct = self.graph.find_dependents(artifact_id, transitive=False)
        transitive = self.graph.find_dependents(artifact_id, transitive=True)
        
        # Calculate max depth
        max_depth = 0
        if transitive:
            for dep in transitive:
                path = self.graph.find_path(artifact_id, dep)
                if len(path) > max_depth:
                    max_depth = len(path) - 1

        return ImpactReport(
            artifact_id=artifact_id,
            direct_dependents=direct,
            transitive_dependents=transitive,
            total_affected=len(transitive),
            max_depth=max_depth,
            tasks_in_flight=[],  # TODO: integrate with task queue
        )

    def root_cause_analysis(self, artifact_id: str) -> RootCauseReport:
        """
        Find root cause of a tainted artifact.
        
        Args:
            artifact_id: Failed/tainted artifact
            
        Returns:
            RootCauseReport with cause chain
        """
        self._load_all_records()
        
        # Find the artifact's provenance
        record = self.get_producer(artifact_id)
        if not record:
            return RootCauseReport(failed_artifact_id=artifact_id)

        # Trace back to find tainted inputs
        dependencies = self.graph.find_dependencies(artifact_id, transitive=True)
        
        tainted_deps = []
        for dep_id in dependencies:
            dep_record = self.get_producer(dep_id)
            if dep_record and dep_record.is_tainted():
                tainted_deps.append((dep_id, dep_record))

        if not tainted_deps:
            # This artifact is the root cause
            return RootCauseReport(
                failed_artifact_id=artifact_id,
                immediate_cause=artifact_id,
                root_cause=artifact_id,
                path=[artifact_id],
                depth=0,
            )

        # Find the earliest tainted artifact (root cause)
        # Sort by cascade depth if available, or by timestamp
        tainted_deps.sort(key=lambda x: x[1].taint.tainted_at or datetime.min)
        root_id, root_record = tainted_deps[0]

        # Find path from root to this artifact
        path = self.graph.find_path(root_id, artifact_id)
        if not path:
            path = [root_id, artifact_id]

        # Immediate cause is the direct dependency
        immediate = record.get_input_artifact_ids()[0] if record.get_input_artifact_ids() else root_id

        return RootCauseReport(
            failed_artifact_id=artifact_id,
            immediate_cause=immediate,
            root_cause=root_id,
            path=path,
            depth=len(path) - 1,
        )

    def consistency_check(self, project_id: str) -> ConsistencyReport:
        """
        Check project consistency and health.
        
        Args:
            project_id: Project to check
            
        Returns:
            ConsistencyReport with status and recommendations
        """
        records = self.get_by_project(project_id)
        
        tainted = sum(1 for r in records if r.is_tainted())
        suspect = sum(1 for r in records if r.is_suspect())
        pending = sum(1 for r in records if r.verification.status == VerificationStatus.PENDING)
        
        # Determine recommended action
        if tainted > len(records) * 0.3:  # More than 30% tainted
            action = "rollback"
        elif suspect > 20 or tainted > 5:
            action = "pause"
        else:
            action = "continue"

        return ConsistencyReport(
            project_id=project_id,
            total_records=len(records),
            tainted_count=tainted,
            suspect_count=suspect,
            pending_revalidation=suspect,  # Suspects need revalidation
            orphaned_artifacts=0,  # TODO: implement orphan detection
            recommended_action=action,
        )

    def get_taint_events(
        self,
        artifact_id: Optional[str] = None,
        cascade_id: Optional[str] = None,
    ) -> list[TaintEvent]:
        """
        Get taint events, optionally filtered.
        
        Args:
            artifact_id: Filter by artifact
            cascade_id: Filter by cascade
            
        Returns:
            List of TaintEvents
        """
        events = self._taint_events
        
        if artifact_id:
            events = [e for e in events if e.artifact_id == artifact_id]
        if cascade_id:
            events = [e for e in events if e.cascade_id == cascade_id]
            
        return events

    # Checkpoint methods

    def create_checkpoint(
        self,
        project_id: str,
        description: str = "",
        trigger: str = "manual",
    ) -> Checkpoint:
        """
        Create a checkpoint for potential rollback.
        
        Args:
            project_id: Project to checkpoint
            description: Optional description
            trigger: What triggered the checkpoint
            
        Returns:
            New Checkpoint
        """
        records = self.get_by_project(project_id)
        
        checkpoint = Checkpoint(
            project_id=project_id,
            provenance_count=len(records),
            last_provenance_id=records[-1].id if records else "",
            last_task_id=records[-1].task_id if records else "",
            description=description,
            trigger=trigger,
        )
        
        self._checkpoints[checkpoint.id] = checkpoint
        
        if self.storage_path:
            self._save_checkpoints()
            
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def get_checkpoints(self, project_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a project."""
        return [c for c in self._checkpoints.values() if c.project_id == project_id]

    def invalidate_checkpoint(self, checkpoint_id: str):
        """Mark a checkpoint as invalid."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint:
            checkpoint.is_valid = False
            if self.storage_path:
                self._save_checkpoints()

    # Storage methods

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

    def _load_all_records(self):
        """Load all records from disk."""
        if not self.storage_path:
            return

        for file_path in self.storage_path.glob("*.json"):
            if file_path.name in ("taint_events.json", "checkpoints.json"):
                continue
            record_id = file_path.stem
            if record_id not in self._records:
                self._load_record(record_id)

    def _save_taint_events(self):
        """Save taint events to disk."""
        if not self.storage_path:
            return

        with open(self._taint_events_path, "w") as f:
            json.dump([e.to_dict() for e in self._taint_events], f, indent=2)

    def _load_taint_events(self):
        """Load taint events from disk."""
        if not self.storage_path or not self._taint_events_path.exists():
            return

        with open(self._taint_events_path) as f:
            data = json.load(f)
            self._taint_events = [TaintEvent.from_dict(e) for e in data]

    def _save_checkpoints(self):
        """Save checkpoints to disk."""
        if not self.storage_path:
            return

        with open(self._checkpoints_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._checkpoints.items()}, f, indent=2)

    def _load_checkpoints(self):
        """Load checkpoints from disk."""
        if not self.storage_path or not self._checkpoints_path.exists():
            return

        with open(self._checkpoints_path) as f:
            data = json.load(f)
            self._checkpoints = {k: Checkpoint.from_dict(v) for k, v in data.items()}


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