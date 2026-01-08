"""
Code General worker for DATS.

Handles general-purpose code generation and modification tasks.
"""

import json
from typing import Any, Optional

from src.workers.base import BaseWorker, WorkerContext


class CodeGeneralWorker(BaseWorker):
    """
    Worker for general code generation tasks.

    Handles standard programming tasks across various languages
    including Python, JavaScript, TypeScript, Go, Rust, etc.
    """

    worker_name = "code_general"
    domain = "code-general"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """Render the code general prompt."""
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
            "dependencies": json.dumps(task_data.get("dependencies", []), indent=2),
            "output_format": task_data.get("output_format", "code"),
        }

        return self._renderer.render_worker("code_general", variables)

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """Get system prompt for code general worker."""
        return """You are an expert software engineer working on a distributed task system.
Your role is to write high-quality, production-ready code.

Guidelines:
1. Follow best practices and coding standards
2. Write clear, well-documented code
3. Consider edge cases and error handling
4. Optimize for readability and maintainability
5. Include necessary imports and dependencies

Output your code in markdown code blocks with appropriate language tags."""

    def _get_temperature(self) -> float:
        """Use moderate temperature for code generation."""
        return 0.5