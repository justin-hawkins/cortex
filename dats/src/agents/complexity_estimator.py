"""
Complexity estimator agent for DATS.

Responsible for estimating task complexity and recommending model tiers.
"""

import json
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext
from src.telemetry import get_tracer

# Get tracer for complexity estimator spans
tracer = get_tracer("dats.complexity_estimator")


class ComplexityEstimator(BaseAgent):
    """
    Complexity estimator agent that analyzes task requirements.

    Responsibilities:
    - Estimate computational complexity
    - Recommend appropriate model tier
    - Estimate token requirements
    - Identify special capabilities needed
    - Suggest quality assurance profile
    """

    agent_name = "complexity_estimator"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the complexity estimator prompt.

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
            "domain": task_data.get("domain", "code-general"),
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "code_context": task_data.get("code_context", ""),
        }

        return self._renderer.render_agent("complexity_estimator", variables)

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """Get system prompt for complexity estimator."""
        return """You are a Complexity Estimator agent in a distributed agentic task system.
Your role is to analyze tasks and estimate their computational requirements.

Available tiers:
- tiny: Simple tasks, basic transformations (gemma3:4b, 32k context)
- small: Moderate tasks, standard coding (gemma3:12b, 32k context)
- large: Complex tasks, advanced reasoning (qwen3-coder/gpt-oss, 64k context)
- frontier: Most complex tasks, requires best models (claude-sonnet-4, 200k context)

Output a JSON object with:
- recommended_tier: string (tiny/small/large/frontier)
- estimated_tokens: int (estimated input tokens)
- confidence: float (0.0-1.0)
- reasoning: string
- required_capabilities: list[string]
- qa_profile: string (consensus/adversarial/security/testing/documentation/human/none)"""

    def _get_temperature(self) -> float:
        """Use low temperature for consistent estimation."""
        return 0.3

    def _process_response(
        self,
        response,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process complexity estimation response.

        Args:
            response: Model response
            task_data: Original task data

        Returns:
            Complexity estimation result
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

        # Default estimation if parsing fails
        return {
            "recommended_tier": "small",
            "estimated_tokens": 5000,
            "confidence": 0.5,
            "reasoning": "Unable to parse response, using default",
            "required_capabilities": [],
            "qa_profile": "consensus",
            "raw_response": content,
        }

    async def estimate(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """
        Estimate complexity of a task.

        Args:
            task_data: Task to estimate

        Returns:
            Complexity estimation
        """
        with tracer.start_as_current_span("estimator.estimate") as span:
            task_id = task_data.get("id", "unknown")
            description = task_data.get("description", "")
            domain = task_data.get("domain", "code-general")

            # Record input event
            span.set_attribute("dats.task.id", task_id)
            span.set_attribute("dats.estimator.domain", domain)
            span.set_attribute("dats.estimator.description_length", len(description))
            span.add_event(
                "estimator.input",
                attributes={
                    "task_id": task_id,
                    "domain": domain,
                    "description_preview": description[:500],  # Truncate for telemetry
                }
            )

            result = await self.execute(task_data)

            if result.success:
                estimation = result.content
                final_result = {
                    "status": "estimated",
                    "task_id": task_id,
                    "recommended_tier": estimation.get("recommended_tier", "small"),
                    "estimated_tokens": estimation.get("estimated_tokens", 5000),
                    "confidence": estimation.get("confidence", 0.5),
                    "reasoning": estimation.get("reasoning", ""),
                    "required_capabilities": estimation.get("required_capabilities", []),
                    "qa_profile": estimation.get("qa_profile", "consensus"),
                    "prompt_version": result.prompt_version,
                }
            else:
                final_result = {
                    "status": "error",
                    "task_id": task_id,
                    "error": result.error,
                    # Default values on error
                    "recommended_tier": "small",
                    "estimated_tokens": 5000,
                    "confidence": 0.0,
                }

            # Record output event with estimation decision
            span.set_attribute("dats.estimator.status", final_result["status"])
            span.set_attribute("dats.estimator.recommended_tier", final_result["recommended_tier"])
            span.set_attribute("dats.estimator.estimated_tokens", final_result["estimated_tokens"])
            span.set_attribute("dats.estimator.confidence", final_result.get("confidence", 0.0))
            span.add_event(
                "estimator.output",
                attributes={
                    "status": final_result["status"],
                    "recommended_tier": final_result["recommended_tier"],
                    "estimated_tokens": str(final_result["estimated_tokens"]),
                    "confidence": str(final_result.get("confidence", 0.0)),
                    "qa_profile": final_result.get("qa_profile", "unknown"),
                    "reasoning": final_result.get("reasoning", "")[:500],  # Truncate
                }
            )

            return final_result
