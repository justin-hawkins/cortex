"""
Lazy revalidation for DATS.

Provides intelligent revalidation of suspect artifacts, determining
whether a flaw in an input actually affected the output.
"""

import asyncio
import heapq
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.storage.provenance import ProvenanceTracker, ProvenanceRecord
from src.storage.work_product import WorkProductStore

logger = logging.getLogger(__name__)


class RevalidationVerdict(str, Enum):
    """Outcome of revalidation evaluation."""

    STILL_VALID = "still_valid"  # Flaw didn't affect output
    INVALID = "invalid"  # Flaw did affect output, escalate to taint
    UNCERTAIN = "uncertain"  # Can't determine, need human or re-run


@dataclass
class RevalidationTask:
    """Task for revalidating a suspect artifact."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # What to revalidate
    suspect_artifact_id: str = ""
    suspect_provenance_id: str = ""
    project_id: str = ""
    
    # Context about the taint
    taint_source_artifact_id: str = ""
    taint_reason: str = ""
    cascade_id: str = ""
    cascade_depth: int = 0
    
    # For queue ordering
    priority: int = 0  # Higher = more urgent
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Parent task for tracking
    parent_revalidation_id: str = ""
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verdict: Optional[RevalidationVerdict] = None
    verdict_reason: str = ""

    def __lt__(self, other):
        """For priority queue ordering (higher priority first, then older first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "suspect_artifact_id": self.suspect_artifact_id,
            "suspect_provenance_id": self.suspect_provenance_id,
            "project_id": self.project_id,
            "taint_source_artifact_id": self.taint_source_artifact_id,
            "taint_reason": self.taint_reason,
            "cascade_id": self.cascade_id,
            "cascade_depth": self.cascade_depth,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "parent_revalidation_id": self.parent_revalidation_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "verdict": self.verdict.value if self.verdict else None,
            "verdict_reason": self.verdict_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RevalidationTask":
        """Create from dictionary."""
        verdict = None
        if data.get("verdict"):
            try:
                verdict = RevalidationVerdict(data["verdict"])
            except ValueError:
                pass
                
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            suspect_artifact_id=data.get("suspect_artifact_id", ""),
            suspect_provenance_id=data.get("suspect_provenance_id", ""),
            project_id=data.get("project_id", ""),
            taint_source_artifact_id=data.get("taint_source_artifact_id", ""),
            taint_reason=data.get("taint_reason", ""),
            cascade_id=data.get("cascade_id", ""),
            cascade_depth=data.get("cascade_depth", 0),
            priority=data.get("priority", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            parent_revalidation_id=data.get("parent_revalidation_id", ""),
            status=data.get("status", "pending"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            verdict=verdict,
            verdict_reason=data.get("verdict_reason", ""),
        )


@dataclass
class RevalidationResult:
    """Result of a revalidation evaluation."""

    task_id: str = ""
    artifact_id: str = ""
    verdict: RevalidationVerdict = RevalidationVerdict.UNCERTAIN
    confidence: float = 0.0
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "details": self.details,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
        }


class RevalidationQueue:
    """
    Priority queue for revalidation tasks.
    
    Manages pending revalidations with configurable ordering
    (depth-first vs breadth-first) and threshold tracking.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_depth: int = 5,
        max_count: int = 50,
        order: str = "depth_first",  # depth_first or breadth_first
    ):
        """
        Initialize revalidation queue.
        
        Args:
            storage_path: Path for persistent queue storage
            max_depth: Maximum cascade depth before recommending rollback
            max_count: Maximum task count before recommending rollback
            order: Queue ordering strategy
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_depth = max_depth
        self.max_count = max_count
        self.order = order
        
        self._pending: list[RevalidationTask] = []
        self._completed: dict[str, RevalidationTask] = {}
        self._by_cascade: dict[str, list[str]] = {}  # cascade_id -> task_ids
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load()

    def add(self, task: RevalidationTask) -> str:
        """
        Add a task to the queue.
        
        Args:
            task: Revalidation task to add
            
        Returns:
            Task ID
        """
        # Adjust priority based on ordering strategy
        if self.order == "depth_first":
            # Higher depth = higher priority (process deeper first)
            task.priority = task.cascade_depth
        else:
            # Lower depth = higher priority (breadth first)
            task.priority = -task.cascade_depth

        heapq.heappush(self._pending, task)
        
        # Track by cascade
        if task.cascade_id:
            if task.cascade_id not in self._by_cascade:
                self._by_cascade[task.cascade_id] = []
            self._by_cascade[task.cascade_id].append(task.id)

        if self.storage_path:
            self._save()

        logger.debug(f"Added revalidation task {task.id} for artifact {task.suspect_artifact_id}")
        return task.id

    def pop(self) -> Optional[RevalidationTask]:
        """
        Get next task from queue.
        
        Returns:
            Next task or None if empty
        """
        if not self._pending:
            return None
            
        task = heapq.heappop(self._pending)
        task.status = "processing"
        task.started_at = datetime.utcnow()
        
        if self.storage_path:
            self._save()
            
        return task

    def complete(self, task: RevalidationTask, verdict: RevalidationVerdict, reason: str = ""):
        """
        Mark a task as completed.
        
        Args:
            task: Task to complete
            verdict: Revalidation verdict
            reason: Reason for verdict
        """
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        task.verdict = verdict
        task.verdict_reason = reason
        
        self._completed[task.id] = task
        
        if self.storage_path:
            self._save()

    def fail(self, task: RevalidationTask, error: str):
        """
        Mark a task as failed.
        
        Args:
            task: Task that failed
            error: Error message
        """
        task.status = "failed"
        task.completed_at = datetime.utcnow()
        task.verdict_reason = f"Error: {error}"
        
        self._completed[task.id] = task
        
        if self.storage_path:
            self._save()

    def get_task(self, task_id: str) -> Optional[RevalidationTask]:
        """Get a task by ID."""
        for task in self._pending:
            if task.id == task_id:
                return task
        return self._completed.get(task_id)

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return len(self._pending)

    def get_cascade_tasks(self, cascade_id: str) -> list[RevalidationTask]:
        """Get all tasks for a cascade."""
        task_ids = self._by_cascade.get(cascade_id, [])
        tasks = []
        for tid in task_ids:
            task = self.get_task(tid)
            if task:
                tasks.append(task)
        return tasks

    def get_max_pending_depth(self) -> int:
        """Get maximum depth among pending tasks."""
        if not self._pending:
            return 0
        return max(t.cascade_depth for t in self._pending)

    def exceeds_thresholds(self) -> bool:
        """Check if queue exceeds configured thresholds."""
        return (
            self.get_pending_count() > self.max_count
            or self.get_max_pending_depth() > self.max_depth
        )

    def clear_cascade(self, cascade_id: str):
        """Clear all tasks for a cascade (e.g., after rollback)."""
        task_ids = set(self._by_cascade.get(cascade_id, []))
        self._pending = [t for t in self._pending if t.id not in task_ids]
        heapq.heapify(self._pending)
        
        for tid in task_ids:
            self._completed.pop(tid, None)
            
        self._by_cascade.pop(cascade_id, None)
        
        if self.storage_path:
            self._save()

    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        completed_verdicts = {}
        for task in self._completed.values():
            if task.verdict:
                v = task.verdict.value
                completed_verdicts[v] = completed_verdicts.get(v, 0) + 1

        return {
            "pending_count": len(self._pending),
            "completed_count": len(self._completed),
            "max_pending_depth": self.get_max_pending_depth(),
            "cascade_count": len(self._by_cascade),
            "exceeds_thresholds": self.exceeds_thresholds(),
            "completed_verdicts": completed_verdicts,
        }

    def _save(self):
        """Save queue to disk."""
        if not self.storage_path:
            return
            
        data = {
            "pending": [t.to_dict() for t in self._pending],
            "completed": {k: v.to_dict() for k, v in self._completed.items()},
            "by_cascade": self._by_cascade,
        }
        
        with open(self.storage_path / "revalidation_queue.json", "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load queue from disk."""
        if not self.storage_path:
            return
            
        queue_file = self.storage_path / "revalidation_queue.json"
        if not queue_file.exists():
            return
            
        with open(queue_file) as f:
            data = json.load(f)
            
        self._pending = [RevalidationTask.from_dict(t) for t in data.get("pending", [])]
        heapq.heapify(self._pending)
        
        self._completed = {
            k: RevalidationTask.from_dict(v)
            for k, v in data.get("completed", {}).items()
        }
        
        self._by_cascade = data.get("by_cascade", {})


class RevalidationEvaluator:
    """
    Evaluate whether a suspect artifact is still valid.
    
    Uses model inference to determine if a flaw in an input artifact
    actually affected the validity of the output artifact.
    """

    def __init__(
        self,
        provenance_tracker: ProvenanceTracker,
        work_product_store: Optional[WorkProductStore] = None,
        model_endpoint: str = "http://192.168.1.11:8000/v1",
        model_name: str = "openai/gpt-oss-20b",
    ):
        """
        Initialize revalidation evaluator.
        
        Args:
            provenance_tracker: For looking up provenance
            work_product_store: For loading artifact content
            model_endpoint: vLLM endpoint for revalidation model
            model_name: Model to use for revalidation
        """
        self.tracker = provenance_tracker
        self.work_product_store = work_product_store
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self._client = None

    async def evaluate(self, task: RevalidationTask) -> RevalidationResult:
        """
        Evaluate whether a suspect artifact is still valid.
        
        Args:
            task: Revalidation task
            
        Returns:
            RevalidationResult with verdict
        """
        start_time = datetime.utcnow()
        result = RevalidationResult(
            task_id=task.id,
            artifact_id=task.suspect_artifact_id,
        )

        try:
            # Load context
            suspect_record = self.tracker.get_record(task.suspect_provenance_id)
            if not suspect_record:
                result.verdict = RevalidationVerdict.UNCERTAIN
                result.reason = "Could not find provenance record"
                return result

            taint_record = self.tracker.get_producer(task.taint_source_artifact_id)
            
            # Get artifact content if available
            suspect_content = self._get_artifact_content(task.suspect_artifact_id)
            taint_content = self._get_artifact_content(task.taint_source_artifact_id)

            # Build revalidation prompt
            prompt = self._build_prompt(
                suspect_record=suspect_record,
                suspect_content=suspect_content,
                taint_reason=task.taint_reason,
                taint_content=taint_content,
            )

            # Call model for evaluation
            response = await self._call_model(prompt)
            
            # Parse response
            verdict, confidence, reason = self._parse_response(response)
            
            result.verdict = verdict
            result.confidence = confidence
            result.reason = reason
            result.details = {
                "prompt_length": len(prompt),
                "response_length": len(response) if response else 0,
            }

        except Exception as e:
            logger.error(f"Revalidation evaluation error: {e}")
            result.verdict = RevalidationVerdict.UNCERTAIN
            result.reason = f"Evaluation error: {e}"

        end_time = datetime.utcnow()
        result.duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return result

    def _get_artifact_content(self, artifact_id: str) -> str:
        """Get content of an artifact."""
        if not self.work_product_store:
            return "[Content not available]"
            
        try:
            artifact = self.work_product_store.get(artifact_id)
            if artifact:
                # Truncate if too long
                content = artifact.content
                if len(content) > 5000:
                    content = content[:5000] + "\n... [truncated]"
                return content
        except Exception as e:
            logger.warning(f"Could not load artifact {artifact_id}: {e}")
            
        return "[Content not available]"

    def _build_prompt(
        self,
        suspect_record: ProvenanceRecord,
        suspect_content: str,
        taint_reason: str,
        taint_content: str,
    ) -> str:
        """Build the revalidation prompt."""
        return f"""You are evaluating whether a flaw in an input artifact affected the validity of an output artifact.

## Tainted Input Artifact
The following input was found to have a flaw:

**Flaw Description:** {taint_reason}

**Input Content:**
```
{taint_content}
```

## Suspect Output Artifact
This output was produced using the tainted input:

**Task:** {suspect_record.task_id}
**Model:** {suspect_record.execution.model}

**Output Content:**
```
{suspect_content}
```

## Question
Given that the input artifact had the flaw described above, is the output artifact still valid?

Consider:
1. Does the output directly incorporate the flawed information?
2. Could the output be correct despite the flawed input?
3. Is the flaw in the input relevant to what the output produces?

## Response Format
Respond with a JSON object:
{{
    "verdict": "STILL_VALID" | "INVALID" | "UNCERTAIN",
    "confidence": <0.0 to 1.0>,
    "reason": "<brief explanation>"
}}

Only respond with the JSON object, no additional text."""

    async def _call_model(self, prompt: str) -> str:
        """Call the revalidation model."""
        try:
            # Lazy import to avoid circular dependencies
            from src.models.openai_client import OpenAICompatibleClient
            
            if self._client is None:
                self._client = OpenAICompatibleClient(
                    endpoint=self.model_endpoint,
                    model_name=self.model_name,
                )
            
            response = await self._client.complete(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
            )
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            raise

    def _parse_response(self, response: str) -> tuple[RevalidationVerdict, float, str]:
        """Parse model response into verdict."""
        if not response:
            return RevalidationVerdict.UNCERTAIN, 0.0, "Empty response"

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or not lines[0].startswith("```"):
                        json_lines.append(line)
                response = "\n".join(json_lines)

            data = json.loads(response)
            
            verdict_str = data.get("verdict", "UNCERTAIN").upper()
            verdict_map = {
                "STILL_VALID": RevalidationVerdict.STILL_VALID,
                "INVALID": RevalidationVerdict.INVALID,
                "UNCERTAIN": RevalidationVerdict.UNCERTAIN,
            }
            verdict = verdict_map.get(verdict_str, RevalidationVerdict.UNCERTAIN)
            
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            reason = data.get("reason", "No reason provided")
            
            return verdict, confidence, reason
            
        except json.JSONDecodeError:
            # Try to parse naturally
            response_lower = response.lower()
            if "still_valid" in response_lower or "still valid" in response_lower:
                return RevalidationVerdict.STILL_VALID, 0.5, "Inferred from response"
            elif "invalid" in response_lower:
                return RevalidationVerdict.INVALID, 0.5, "Inferred from response"
            else:
                return RevalidationVerdict.UNCERTAIN, 0.3, "Could not parse response"

    async def close(self):
        """Clean up resources."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()


async def process_revalidation_queue(
    queue: RevalidationQueue,
    evaluator: RevalidationEvaluator,
    taint_propagator: "TaintPropagator",  # Forward reference
    max_batch: int = 10,
) -> dict[str, Any]:
    """
    Process pending revalidation tasks.
    
    Args:
        queue: Revalidation queue
        evaluator: Evaluator for revalidation
        taint_propagator: For escalating invalid artifacts
        max_batch: Maximum tasks to process
        
    Returns:
        Processing statistics
    """
    from src.cascade.taint import TaintPropagator
    
    processed = 0
    still_valid = 0
    invalid = 0
    uncertain = 0
    errors = 0

    for _ in range(max_batch):
        task = queue.pop()
        if not task:
            break

        try:
            result = await evaluator.evaluate(task)
            
            if result.verdict == RevalidationVerdict.STILL_VALID:
                # Clear suspect status, stop cascade on this branch
                queue.complete(task, result.verdict, result.reason)
                taint_propagator.clear_suspect_status(
                    task.suspect_artifact_id,
                    reason=result.reason,
                )
                still_valid += 1
                
            elif result.verdict == RevalidationVerdict.INVALID:
                # Escalate to taint, continue cascade
                queue.complete(task, result.verdict, result.reason)
                taint_propagator.escalate_suspect_to_taint(
                    task.suspect_artifact_id,
                    reason=result.reason,
                    cascade_id=task.cascade_id,
                )
                invalid += 1
                
            else:  # UNCERTAIN
                # Mark for human review or re-execution
                queue.complete(task, result.verdict, result.reason)
                # TODO: Queue for human review
                uncertain += 1

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process revalidation task {task.id}: {e}")
            queue.fail(task, str(e))
            errors += 1

    return {
        "processed": processed,
        "still_valid": still_valid,
        "invalid": invalid,
        "uncertain": uncertain,
        "errors": errors,
        "remaining": queue.get_pending_count(),
    }