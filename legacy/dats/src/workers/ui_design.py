"""
UI Design worker for DATS.

Handles UI/UX design and frontend development tasks.
"""

import json
from typing import Any, Optional

from src.workers.base import BaseWorker, WorkerContext


class UIDesignWorker(BaseWorker):
    """
    Worker for UI/UX design tasks.

    Handles frontend development, component design,
    styling, and user interface implementation.
    """

    worker_name = "ui_design"
    domain = "ui-design"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """Render the UI design prompt."""
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
            "ui_framework": task_data.get("ui_framework", "react"),
            "design_system": task_data.get("design_system", ""),
        }

        return self._renderer.render_worker("ui_design", variables)

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """Get system prompt for UI design worker."""
        return """You are an expert frontend developer and UI/UX designer.
Your role is to create beautiful, accessible, and responsive user interfaces.

Specializations:
1. React, Vue, and modern frontend frameworks
2. CSS, Tailwind, and styling systems
3. Component architecture and design systems
4. Accessibility (WCAG compliance)
5. Responsive and mobile-first design

Output your code in markdown code blocks with appropriate language tags."""

    def _get_temperature(self) -> float:
        """Use moderate temperature for UI design (allows creativity)."""
        return 0.6