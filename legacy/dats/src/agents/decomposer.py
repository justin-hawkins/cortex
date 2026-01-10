"""
Decomposer agent for DATS.

Responsible for breaking down complex tasks into smaller subtasks.
"""

import json
import uuid
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext


# Atomicity detection thresholds
ATOMICITY_THRESHOLDS = {
    # Maximum token estimate for atomic tasks
    "max_tokens": 8000,
    # Single-domain indicators
    "single_domain_keywords": {
        "code-general": ["function", "class", "method", "module", "script"],
        "documentation": ["readme", "docs", "guide", "tutorial", "api docs"],
        "ui-design": ["component", "page", "layout", "style", "theme"],
        "code-embedded": ["driver", "interrupt", "register", "firmware"],
        "code-vision": ["filter", "detection", "recognition", "transform"],
    },
    # Complexity keywords that suggest non-atomic
    "non_atomic_keywords": [
        "multiple", "several", "system", "integrate",
        "full stack", "complete", "end-to-end",
    ],
}


class Decomposer(BaseAgent):
    """
    Decomposer agent that breaks down complex tasks.

    Responsibilities:
    - Analyze task complexity
    - Identify independent work units
    - Define subtask boundaries
    - Specify dependencies between subtasks
    - Recommend worker types for each subtask
    """

    agent_name = "decomposer"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the decomposer prompt.

        Args:
            task_data: Task configuration
            context: Execution context

        Returns:
            Tuple of (rendered_prompt, version_hash)
        """
        variables = {
            "task_id": context.task_id,
            "project_id": context.project_id,
            "task_description": task_data.get("description", ""),
            "inputs": json.dumps(task_data.get("inputs", []), indent=2),
            "acceptance_criteria": task_data.get("acceptance_criteria", ""),
            "domain": task_data.get("domain", "code-general"),
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "lightrag_context": task_data.get("lightrag_context", ""),
            "constitution": task_data.get("constitution", ""),
            "code_context": task_data.get("code_context", ""),
        }

        return self._renderer.render_agent("decomposer", variables)

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """Get system prompt for decomposer."""
        return """You are a Decomposer agent in a distributed agentic task system.
Your role is to break down complex tasks into smaller, manageable subtasks.

Guidelines:
1. Each subtask should be independently executable
2. Minimize dependencies between subtasks
3. Estimate complexity for each subtask (tiny/small/large/frontier)
4. Specify clear inputs and outputs for each subtask
5. Consider parallelization opportunities

Always respond in structured JSON format with a list of subtasks."""

    def _get_temperature(self) -> float:
        """Use moderate temperature for creative decomposition."""
        return 0.6

    def _process_response(
        self,
        response,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process decomposer response.

        Extracts structured subtask definitions.

        Args:
            response: Model response
            task_data: Original task data

        Returns:
            Decomposition result with subtasks
        """
        content = response.content

        # Try to parse as JSON
        try:
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
                return json.loads(json_str)
            elif content.strip().startswith("{"):
                return json.loads(content)
            elif content.strip().startswith("["):
                return {"subtasks": json.loads(content)}
        except json.JSONDecodeError:
            pass

        # Return as raw content if not JSON
        return {
            "subtasks": [],
            "raw_response": content,
        }

    async def decompose(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """
        Decompose a task into subtasks.

        Args:
            task_data: Task to decompose

        Returns:
            Decomposition result with subtasks
        """
        result = await self.execute(task_data)

        if result.success:
            decomposition = result.content
            return {
                "status": "decomposed",
                "parent_task_id": task_data.get("id"),
                "subtasks": decomposition.get("subtasks", []),
                "dependencies": decomposition.get("dependencies", []),
                "prompt_version": result.prompt_version,
            }
        else:
            return {
                "status": "error",
                "parent_task_id": task_data.get("id"),
                "error": result.error,
            }

    async def decompose_recursive(
        self,
        task_data: dict[str, Any],
        max_depth: int = 5,
        current_depth: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Recursively decompose a task until all subtasks are atomic.

        This method handles the full decomposition flow, breaking down
        complex tasks into worker-executable atomic units.

        Args:
            task_data: Task to decompose
            max_depth: Maximum recursion depth (safety limit)
            current_depth: Current recursion depth

        Returns:
            List of atomic subtasks ready for execution
        """
        import logging
        logger = logging.getLogger(__name__)

        # Safety check: max depth
        if current_depth >= max_depth:
            logger.warning(
                f"Max decomposition depth ({max_depth}) reached for task {task_data.get('id')}"
            )
            # Force atomic - mark as needing single worker execution
            task_data["is_atomic"] = True
            task_data["forced_atomic"] = True
            return [task_data]

        # Check if already atomic
        if self.is_atomic(task_data):
            logger.debug(f"Task {task_data.get('id')} is atomic")
            task_data["is_atomic"] = True
            return [task_data]

        # Decompose via LLM
        result = await self.decompose(task_data)

        if result.get("status") != "decomposed":
            logger.warning(
                f"Decomposition failed for {task_data.get('id')}: {result.get('error')}"
            )
            # Treat as atomic on error
            task_data["is_atomic"] = True
            task_data["decomposition_failed"] = True
            return [task_data]

        subtasks = result.get("subtasks", [])

        if not subtasks:
            # No subtasks means already atomic
            logger.debug(f"Task {task_data.get('id')} decomposed to empty - treating as atomic")
            task_data["is_atomic"] = True
            return [task_data]

        # Process each subtask
        atomic_tasks = []
        for i, subtask in enumerate(subtasks):
            # Ensure subtask has required fields
            subtask["id"] = subtask.get("id", str(uuid.uuid4()))
            subtask["parent_task_id"] = task_data.get("id")
            subtask["project_id"] = task_data.get("project_id")
            subtask["sequence"] = i

            # Inherit domain if not specified
            if "domain" not in subtask:
                subtask["domain"] = task_data.get("domain", "code-general")

            # Check if this subtask is atomic
            if self.is_atomic(subtask):
                subtask["is_atomic"] = True
                atomic_tasks.append(subtask)
            else:
                # Recursively decompose
                sub_atomic = await self.decompose_recursive(
                    subtask,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
                atomic_tasks.extend(sub_atomic)

        # Process dependencies if provided
        dependencies = result.get("dependencies", [])
        if dependencies:
            atomic_tasks = self._apply_dependencies(atomic_tasks, dependencies)

        logger.info(
            f"Decomposed task {task_data.get('id')} into {len(atomic_tasks)} atomic tasks"
        )

        return atomic_tasks

    def is_atomic(self, task_data: dict[str, Any]) -> bool:
        """
        Check if a task is atomic (single worker executable).

        A task is considered atomic if it can be handled by a single
        worker in a single execution. This is determined by:
        - Explicit marking
        - Token/size limits
        - Single domain focus
        - Absence of sub-components

        Args:
            task_data: Task to evaluate

        Returns:
            True if task is atomic
        """
        # Explicit marking
        if task_data.get("is_atomic", False):
            return True

        if task_data.get("needs_decomposition", False):
            return False

        # Check complexity level
        complexity = task_data.get("complexity", task_data.get("estimated_complexity", "medium"))
        if complexity in ["tiny", "small"]:
            return True

        # Check estimated tokens
        estimated_tokens = task_data.get("estimated_tokens", 0)
        if estimated_tokens > ATOMICITY_THRESHOLDS["max_tokens"]:
            return False

        # Check for non-atomic keywords in description
        description = task_data.get("description", "").lower()
        for keyword in ATOMICITY_THRESHOLDS["non_atomic_keywords"]:
            if keyword in description:
                return False

        # Check if subtasks are already defined
        if task_data.get("subtasks"):
            return False

        # Check description length as proxy for complexity
        if len(description) > 500:
            return False

        # Check for multiple distinct requirements
        sentences = [s.strip() for s in description.split(".") if len(s.strip()) > 20]
        if len(sentences) > 3:
            return False

        # Default: assume atomic for simple descriptions
        return True

    def estimate_atomicity_score(self, task_data: dict[str, Any]) -> float:
        """
        Calculate an atomicity score for a task.

        Higher score = more likely to be atomic.

        Args:
            task_data: Task to evaluate

        Returns:
            Score from 0.0 (definitely not atomic) to 1.0 (definitely atomic)
        """
        score = 0.5  # Start neutral

        description = task_data.get("description", "").lower()

        # Explicit markers
        if task_data.get("is_atomic"):
            return 1.0
        if task_data.get("needs_decomposition"):
            return 0.0

        # Complexity adjustments
        complexity = task_data.get("complexity", "medium")
        complexity_scores = {
            "tiny": 0.4,
            "small": 0.3,
            "medium": 0.0,
            "large": -0.3,
            "frontier": -0.4,
        }
        score += complexity_scores.get(complexity, 0.0)

        # Token estimate adjustment
        tokens = task_data.get("estimated_tokens", 0)
        if tokens > ATOMICITY_THRESHOLDS["max_tokens"]:
            score -= 0.3
        elif tokens < 2000:
            score += 0.2

        # Description length adjustment
        desc_len = len(description)
        if desc_len < 100:
            score += 0.2
        elif desc_len > 500:
            score -= 0.2

        # Keyword adjustments
        for keyword in ATOMICITY_THRESHOLDS["non_atomic_keywords"]:
            if keyword in description:
                score -= 0.1

        # Single domain keywords boost
        domain = task_data.get("domain", "code-general")
        domain_keywords = ATOMICITY_THRESHOLDS["single_domain_keywords"].get(domain, [])
        for keyword in domain_keywords:
            if keyword in description:
                score += 0.05

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _apply_dependencies(
        self,
        tasks: list[dict[str, Any]],
        dependencies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Apply dependency information to atomic tasks.

        Args:
            tasks: List of atomic tasks
            dependencies: Dependency specifications from decomposition

        Returns:
            Tasks with dependency information added
        """
        # Build task ID map
        task_map = {t.get("id"): t for t in tasks}

        for dep in dependencies:
            from_id = dep.get("from")
            to_id = dep.get("to")

            if from_id in task_map and to_id in task_map:
                # Add dependency to target task
                if "depends_on" not in task_map[to_id]:
                    task_map[to_id]["depends_on"] = []
                task_map[to_id]["depends_on"].append(from_id)

        return list(task_map.values())

    def create_subtask(
        self,
        parent_task: dict[str, Any],
        description: str,
        domain: Optional[str] = None,
        complexity: str = "small",
    ) -> dict[str, Any]:
        """
        Create a new subtask from a parent task.

        Helper method for programmatic subtask creation.

        Args:
            parent_task: Parent task data
            description: Subtask description
            domain: Domain (defaults to parent's domain)
            complexity: Complexity estimate

        Returns:
            New subtask dictionary
        """
        return {
            "id": str(uuid.uuid4()),
            "parent_task_id": parent_task.get("id"),
            "project_id": parent_task.get("project_id"),
            "description": description,
            "domain": domain or parent_task.get("domain", "code-general"),
            "complexity": complexity,
            "is_atomic": True,
            "inputs": [],
        }
