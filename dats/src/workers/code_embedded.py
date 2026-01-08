"""
Code Embedded worker for DATS.

Handles embedded systems and low-level programming tasks.
"""

import json
from typing import Any, Optional

from src.workers.base import BaseWorker, WorkerContext


class CodeEmbeddedWorker(BaseWorker):
    """
    Worker for embedded systems code tasks.

    Specializes in low-level programming, embedded systems,
    hardware interfaces, and resource-constrained environments.
    """

    worker_name = "code_embedded"
    domain = "code-embedded"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """Render the code embedded prompt."""
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
            "target_platform": task_data.get("target_platform", "generic"),
            "memory_constraints": task_data.get("memory_constraints", ""),
        }

        return self._renderer.render_worker("code_embedded", variables)

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """Get system prompt for code embedded worker."""
        return """You are an expert embedded systems engineer.
Your role is to write efficient, resource-aware code for constrained environments.

Specializations:
1. C and C++ for embedded systems
2. Microcontroller programming (ARM, AVR, ESP32)
3. Real-time operating systems (FreeRTOS, Zephyr)
4. Hardware abstraction layers
5. Memory and power optimization

Output your code in markdown code blocks with appropriate language tags."""

    def _get_temperature(self) -> float:
        """Use lower temperature for embedded code (precision matters)."""
        return 0.4