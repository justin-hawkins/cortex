"""
Coordinator agent for DATS.

Responsible for overall task orchestration and workflow management.
"""

import json
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext


# Complexity thresholds for decomposition decisions
DECOMPOSITION_THRESHOLDS = {
    "max_description_length": 500,  # Characters
    "keywords_complex": [
        "multiple", "several", "and also", "as well as",
        "integrate", "system", "full", "complete",
        "including", "with support for",
    ],
    "keywords_simple": [
        "simple", "basic", "single", "just", "only",
        "quick", "small", "trivial",
    ],
}


class Coordinator(BaseAgent):
    """
    Coordinator agent that orchestrates the overall workflow.

    Responsibilities:
    - Receive high-level tasks from users
    - Determine if decomposition is needed
    - Route tasks to appropriate workers
    - Track overall progress
    - Handle escalation and human approval
    """

    agent_name = "coordinator"

    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the coordinator prompt.

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
            "model_name": context.model_name,
            "model_tier": context.model_tier,
            "context_window": context.context_window,
            "lightrag_context": task_data.get("lightrag_context", ""),
            "constitution": task_data.get("constitution", ""),
        }

        return self._renderer.render_agent("coordinator", variables)

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """Get system prompt for coordinator."""
        return """You are a Coordinator agent in a distributed agentic task system.
Your role is to orchestrate complex software development tasks by:
1. Analyzing incoming requests
2. Determining if tasks need decomposition
3. Routing work to appropriate workers
4. Ensuring quality and consistency

Always respond in structured JSON format when creating task plans."""

    def _get_temperature(self) -> float:
        """Use lower temperature for more consistent coordination."""
        return 0.5

    def _process_response(
        self,
        response,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process coordinator response.

        Attempts to parse JSON response for structured output.

        Args:
            response: Model response
            task_data: Original task data

        Returns:
            Processed coordination decision
        """
        content = response.content

        # Try to parse as JSON
        try:
            # Look for JSON in the response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
                return json.loads(json_str)
            elif content.strip().startswith("{"):
                return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Return as raw content if not JSON
        return {
            "decision": "unknown",
            "raw_response": content,
        }

    async def analyze_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze a task to determine next steps.

        Args:
            task_data: Task to analyze

        Returns:
            Analysis result with recommended action
        """
        result = await self.execute(task_data)

        if result.success:
            return {
                "status": "analyzed",
                "recommendation": result.content,
                "prompt_version": result.prompt_version,
            }
        else:
            return {
                "status": "error",
                "error": result.error,
            }

    async def process_request(
        self,
        user_request: str,
        project_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Process a user request and determine the appropriate action.

        This is the main entry point for the coordinator. It analyzes
        the request and returns structured data for the pipeline.

        Args:
            user_request: Natural language request from user
            project_id: Project identifier
            context: Optional additional context

        Returns:
            Dictionary with:
            - mode: Task mode (new_project, modify, fix_bug, etc.)
            - domain: Target domain (code-general, documentation, etc.)
            - needs_decomposition: Whether task needs breakdown
            - complexity: Estimated complexity (tiny/small/medium/large/frontier)
            - qa_profile: Recommended QA profile
            - acceptance_criteria: Extracted or generated acceptance criteria
        """
        # Quick heuristic analysis before LLM call
        quick_analysis = self._quick_analyze(user_request)

        task_data = {
            "description": user_request,
            "project_id": project_id,
            "inputs": [],
        }

        if context:
            task_data["inputs"].append({
                "type": "context",
                "content": context,
            })

        # Call LLM for detailed analysis
        result = await self.analyze_task(task_data)

        if result.get("status") == "analyzed":
            recommendation = result.get("recommendation", {})

            # Merge quick analysis with LLM recommendation
            return {
                "success": True,
                "mode": recommendation.get("mode", quick_analysis["mode"]),
                "domain": recommendation.get("domain", quick_analysis["domain"]),
                "needs_decomposition": recommendation.get(
                    "needs_decomposition",
                    quick_analysis["needs_decomposition"],
                ),
                "complexity": recommendation.get("complexity", quick_analysis["complexity"]),
                "qa_profile": self._determine_qa_profile(
                    recommendation.get("complexity", quick_analysis["complexity"]),
                    recommendation.get("domain", quick_analysis["domain"]),
                ),
                "acceptance_criteria": recommendation.get("acceptance_criteria", ""),
            }
        else:
            # Fall back to quick analysis on error
            return {
                "success": True,
                "mode": quick_analysis["mode"],
                "domain": quick_analysis["domain"],
                "needs_decomposition": quick_analysis["needs_decomposition"],
                "complexity": quick_analysis["complexity"],
                "qa_profile": self._determine_qa_profile(
                    quick_analysis["complexity"],
                    quick_analysis["domain"],
                ),
                "acceptance_criteria": "",
                "warning": f"LLM analysis failed: {result.get('error')}. Using heuristics.",
            }

    def _quick_analyze(self, request: str) -> dict[str, Any]:
        """
        Quick heuristic analysis of a request.

        This provides fast initial analysis before LLM call,
        and serves as fallback if LLM fails.

        Args:
            request: User request text

        Returns:
            Quick analysis results
        """
        request_lower = request.lower()

        # Determine mode
        mode = "new_project"
        if any(word in request_lower for word in ["fix", "bug", "error", "issue", "broken"]):
            mode = "fix_bug"
        elif any(word in request_lower for word in ["update", "modify", "change", "edit", "add to"]):
            mode = "modify"
        elif any(word in request_lower for word in ["refactor", "improve", "optimize", "clean"]):
            mode = "refactor"
        elif any(word in request_lower for word in ["document", "readme", "docs", "explain"]):
            mode = "documentation"
        elif any(word in request_lower for word in ["test", "spec", "coverage"]):
            mode = "testing"

        # Determine domain
        domain = "code-general"
        if any(word in request_lower for word in ["ui", "interface", "frontend", "design", "css", "html"]):
            domain = "ui-design"
        elif any(word in request_lower for word in ["document", "readme", "docs", "markdown"]):
            domain = "documentation"
        elif any(word in request_lower for word in ["embedded", "microcontroller", "arduino", "hardware"]):
            domain = "code-embedded"
        elif any(word in request_lower for word in ["vision", "image", "opencv", "camera"]):
            domain = "code-vision"

        # Check for decomposition need
        needs_decomposition = self._should_decompose(request)

        # Estimate complexity
        complexity = self._estimate_quick_complexity(request)

        return {
            "mode": mode,
            "domain": domain,
            "needs_decomposition": needs_decomposition,
            "complexity": complexity,
        }

    def _should_decompose(self, request: str) -> bool:
        """
        Determine if a request needs decomposition.

        Args:
            request: User request text

        Returns:
            True if decomposition is recommended
        """
        request_lower = request.lower()

        # Check length
        if len(request) > DECOMPOSITION_THRESHOLDS["max_description_length"]:
            return True

        # Check for complexity keywords
        complex_count = sum(
            1 for kw in DECOMPOSITION_THRESHOLDS["keywords_complex"]
            if kw in request_lower
        )
        simple_count = sum(
            1 for kw in DECOMPOSITION_THRESHOLDS["keywords_simple"]
            if kw in request_lower
        )

        # If more complex keywords than simple, suggest decomposition
        if complex_count > simple_count and complex_count >= 2:
            return True

        # Check for multiple distinct requirements (sentences or bullet points)
        sentences = [s.strip() for s in request.split(".") if len(s.strip()) > 20]
        if len(sentences) > 3:
            return True

        return False

    def _estimate_quick_complexity(self, request: str) -> str:
        """
        Quick complexity estimation based on heuristics.

        Args:
            request: User request text

        Returns:
            Complexity tier: tiny, small, medium, large, or frontier
        """
        request_lower = request.lower()

        # Check for simple indicators
        if any(kw in request_lower for kw in DECOMPOSITION_THRESHOLDS["keywords_simple"]):
            if len(request) < 100:
                return "tiny"
            return "small"

        # Check for complex indicators
        complex_count = sum(
            1 for kw in DECOMPOSITION_THRESHOLDS["keywords_complex"]
            if kw in request_lower
        )

        if complex_count >= 3:
            return "large"
        elif complex_count >= 1:
            return "medium"

        # Default based on length
        if len(request) < 50:
            return "tiny"
        elif len(request) < 150:
            return "small"
        elif len(request) < 400:
            return "medium"
        else:
            return "large"

    def _determine_qa_profile(self, complexity: str, domain: str) -> str:
        """
        Determine appropriate QA profile based on task characteristics.

        Args:
            complexity: Task complexity tier
            domain: Task domain

        Returns:
            QA profile name
        """
        # Simple tasks get minimal QA
        if complexity in ["tiny", "small"]:
            return "consensus"

        # Documentation gets documentation-specific QA
        if domain == "documentation":
            return "documentation"

        # Large tasks get more thorough QA
        if complexity in ["large", "frontier"]:
            return "adversarial"

        # Default
        return "consensus"
