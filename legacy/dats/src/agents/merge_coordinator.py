"""
Merge Coordinator agent for DATS.

Responsible for merging results from parallel subtasks.
"""

import json
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext


class MergeCoordinator(BaseAgent):
    """
    Merge Coordinator agent that combines results from subtasks.

    Responsibilities:
    - Collect results from parallel subtasks
    - Resolve conflicts between outputs
    - Integrate code changes
    - Ensure consistency across merged output
    - Handle merge failures gracefully
    """

    agent_name = "merge_coordinator"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the merge coordinator prompt.

        Args:
            task_data: Task configuration including subtask results
            context: Execution context

        Returns:
            Tuple of (rendered_prompt, version_hash)
        """
        variables = {
            "task_id": context.task_id,
            "project_id": context.project_id,
            "parent_task_description": task_data.get("description", ""),
            "subtask_results": json.dumps(
                task_data.get("subtask_results", []), indent=2
            ),
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "merge_strategy": task_data.get("merge_strategy", "auto"),
            "conflicts": json.dumps(task_data.get("conflicts", []), indent=2),
        }

        return self._renderer.render_agent("merge_coordinator", variables)

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """Get system prompt for merge coordinator."""
        return """You are a Merge Coordinator agent in a distributed agentic task system.
Your role is to combine results from parallel subtasks into a coherent output.

Merge guidelines:
1. Identify overlapping changes
2. Resolve conflicts intelligently
3. Ensure code compiles/works after merge
4. Maintain consistency in style and approach
5. Document any manual resolution needed

Output a JSON object with:
- success: boolean
- merged_output: object with combined results
- conflicts_resolved: list of resolved conflicts
- conflicts_remaining: list of unresolved conflicts
- merge_notes: string with important notes
- requires_human_review: boolean"""

    def _get_temperature(self) -> float:
        """Use low temperature for consistent merging."""
        return 0.4

    def _process_response(
        self,
        response,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process merge coordinator response.

        Args:
            response: Model response
            task_data: Original task data

        Returns:
            Merge result
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
        except json.JSONDecodeError:
            pass

        # Default merge result if parsing fails
        return {
            "success": False,
            "merged_output": {},
            "conflicts_resolved": [],
            "conflicts_remaining": [
                {"type": "parse_error", "description": "Unable to parse merge response"}
            ],
            "merge_notes": "Merge response parsing failed",
            "requires_human_review": True,
            "raw_response": content,
        }

    async def merge(
        self,
        parent_task: dict[str, Any],
        subtask_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Merge results from subtasks.

        Args:
            parent_task: Original parent task configuration
            subtask_results: Results from completed subtasks

        Returns:
            Merge result
        """
        # Prepare merge data
        merge_data = {
            **parent_task,
            "subtask_results": subtask_results,
        }

        # Detect potential conflicts
        conflicts = self._detect_conflicts(subtask_results)
        if conflicts:
            merge_data["conflicts"] = conflicts

        result = await self.execute(merge_data)

        if result.success:
            merge_result = result.content
            return {
                "status": "merged",
                "parent_task_id": parent_task.get("id"),
                "success": merge_result.get("success", False),
                "merged_output": merge_result.get("merged_output", {}),
                "conflicts_resolved": merge_result.get("conflicts_resolved", []),
                "conflicts_remaining": merge_result.get("conflicts_remaining", []),
                "merge_notes": merge_result.get("merge_notes", ""),
                "requires_human_review": merge_result.get("requires_human_review", False),
                "prompt_version": result.prompt_version,
            }
        else:
            return {
                "status": "error",
                "parent_task_id": parent_task.get("id"),
                "error": result.error,
                "success": False,
                "requires_human_review": True,
            }

    def _detect_conflicts(
        self, subtask_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Detect potential conflicts between subtask results.

        Args:
            subtask_results: Results to check for conflicts

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for overlapping file modifications
        files_modified = {}
        for i, result in enumerate(subtask_results):
            output = result.get("output", {})
            artifacts = output.get("artifacts", [])
            
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    path = artifact.get("path", "")
                    if path:
                        if path in files_modified:
                            conflicts.append({
                                "type": "file_conflict",
                                "path": path,
                                "subtasks": [files_modified[path], i],
                            })
                        else:
                            files_modified[path] = i

        return conflicts