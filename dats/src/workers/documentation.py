"""
Documentation worker for DATS.

Handles documentation generation and technical writing tasks.
"""

import json
from typing import Any, Optional

from src.workers.base import BaseWorker, WorkerContext


class DocumentationWorker(BaseWorker):
    """
    Worker for documentation tasks.

    Handles technical documentation, API docs, README files,
    user guides, and other written content.
    """

    worker_name = "documentation"
    domain = "documentation"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """Render the documentation prompt."""
        variables = {
            "task_id": context.task_id,
            "project_id": context.project_id,
            "task_description": task_data.get("description", ""),
            "inputs": json.dumps(task_data.get("inputs", []), indent=2),
            "acceptance_criteria": task_data.get("acceptance_criteria", ""),
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "code_context": task_data.get("code_context", ""),
            "doc_type": task_data.get("doc_type", "technical"),
            "audience": task_data.get("audience", "developers"),
        }

        return self._renderer.render_worker("documentation", variables)

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """Get system prompt for documentation worker."""
        return """You are an expert technical writer.
Your role is to create clear, comprehensive documentation.

Guidelines:
1. Write for the target audience
2. Use clear, concise language
3. Include relevant examples
4. Structure content logically
5. Use appropriate formatting (markdown)

Output documentation in well-structured markdown format."""

    def _get_temperature(self) -> float:
        """Use moderate temperature for documentation."""
        return 0.6

    def _get_max_tokens(self, context: WorkerContext) -> int:
        """Allow more tokens for documentation output."""
        return min(12000, context.safe_working_limit // 2)