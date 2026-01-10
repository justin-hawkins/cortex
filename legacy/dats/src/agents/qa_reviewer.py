"""
QA Reviewer agent for DATS.

Responsible for validating task outputs and ensuring quality.
"""

import json
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext


class QAReviewer(BaseAgent):
    """
    QA Reviewer agent that validates task outputs.

    Responsibilities:
    - Review code for correctness
    - Check against acceptance criteria
    - Identify potential issues
    - Provide feedback for improvements
    - Approve or reject work products
    """

    agent_name = "qa_reviewer"

    def __init__(
        self,
        model_client=None,
        model_tier: Optional[str] = None,
        adversarial: bool = False,
    ):
        """
        Initialize QA Reviewer.

        Args:
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
            adversarial: Use adversarial review mode
        """
        super().__init__(model_client, model_tier)
        self.adversarial = adversarial
        if adversarial:
            self.agent_name = "qa_reviewer_adversarial"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the QA reviewer prompt.

        Args:
            task_data: Task configuration including work product to review
            context: Execution context

        Returns:
            Tuple of (rendered_prompt, version_hash)
        """
        variables = {
            "task_id": context.task_id,
            "project_id": context.project_id,
            "task_description": task_data.get("description", ""),
            "acceptance_criteria": task_data.get("acceptance_criteria", ""),
            "work_product": task_data.get("work_product", ""),
            "domain": task_data.get("domain", "code-general"),
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "qa_profile": task_data.get("qa_profile", "consensus"),
            "validation_checks": json.dumps(
                task_data.get("validation_checks", []), indent=2
            ),
        }

        return self._renderer.render_agent("qa_reviewer", variables)

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """Get system prompt for QA reviewer."""
        base_prompt = """You are a QA Reviewer agent in a distributed agentic task system.
Your role is to validate work products and ensure they meet quality standards.

Review criteria:
1. Correctness: Does the output match the requirements?
2. Completeness: Are all acceptance criteria met?
3. Quality: Is the code/output well-structured?
4. Security: Are there any security concerns?
5. Performance: Are there any performance issues?

Output a JSON object with:
- approved: boolean
- confidence: float (0.0-1.0)
- issues: list of {severity: string, description: string, location: string}
- suggestions: list of strings
- summary: string"""

        if self.adversarial:
            base_prompt += """

ADVERSARIAL MODE: Be extra critical. Actively look for edge cases, 
potential failures, and subtle bugs. Challenge assumptions."""

        return base_prompt

    def _get_temperature(self) -> float:
        """Use low temperature for consistent review."""
        return 0.3 if not self.adversarial else 0.5

    def _process_response(
        self,
        response,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process QA review response.

        Args:
            response: Model response
            task_data: Original task data

        Returns:
            Review result
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

        # Default review if parsing fails
        return {
            "approved": False,
            "confidence": 0.0,
            "issues": [
                {
                    "severity": "error",
                    "description": "Unable to parse review response",
                    "location": "unknown",
                }
            ],
            "suggestions": [],
            "summary": "Review parsing failed",
            "raw_response": content,
        }

    async def review(
        self,
        task_data: dict[str, Any],
        work_product: str,
    ) -> dict[str, Any]:
        """
        Review a work product.

        Args:
            task_data: Original task configuration
            work_product: The output to review

        Returns:
            Review result
        """
        # Add work product to task data for prompt rendering
        review_data = {**task_data, "work_product": work_product}

        result = await self.execute(review_data)

        if result.success:
            review = result.content
            return {
                "status": "reviewed",
                "task_id": task_data.get("id"),
                "approved": review.get("approved", False),
                "confidence": review.get("confidence", 0.0),
                "issues": review.get("issues", []),
                "suggestions": review.get("suggestions", []),
                "summary": review.get("summary", ""),
                "prompt_version": result.prompt_version,
                "adversarial": self.adversarial,
            }
        else:
            return {
                "status": "error",
                "task_id": task_data.get("id"),
                "error": result.error,
                "approved": False,
            }