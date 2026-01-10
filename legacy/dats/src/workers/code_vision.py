"""
Code Vision worker for DATS.

Handles computer vision and ML/AI-related code tasks.
"""

import json
from typing import Any, Optional

from src.workers.base import BaseWorker, WorkerContext


class CodeVisionWorker(BaseWorker):
    """
    Worker for computer vision and ML code tasks.

    Specializes in machine learning, deep learning, computer vision,
    and AI-related code generation and modification.
    """

    worker_name = "code_vision"
    domain = "code-vision"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """Render the code vision prompt."""
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
            "ml_framework": task_data.get("ml_framework", "pytorch"),
        }

        return self._renderer.render_worker("code_vision", variables)

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """Get system prompt for code vision worker."""
        return """You are an expert ML/AI engineer specializing in computer vision.
Your role is to write production-quality ML code.

Specializations:
1. PyTorch and TensorFlow frameworks
2. Computer vision models (CNN, Vision Transformers)
3. Image processing and augmentation
4. Model training, evaluation, and inference
5. MLOps and model deployment

Output your code in markdown code blocks with appropriate language tags."""

    def _get_temperature(self) -> float:
        """Use moderate temperature for ML code generation."""
        return 0.5