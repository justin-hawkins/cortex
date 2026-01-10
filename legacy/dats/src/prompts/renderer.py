"""
Prompt template renderer for DATS.

Handles variable substitution in prompt templates.
"""

import re
from typing import Any, Optional

from src.prompts.loader import LoadedPrompt, get_prompt_loader


class PromptRenderer:
    """
    Render prompt templates with variable substitution.

    Supports {variable_name} syntax for dynamic content injection.
    """

    # Standard variables that can be injected into prompts
    STANDARD_VARIABLES = {
        "task_id",
        "task_description",
        "inputs",
        "acceptance_criteria",
        "lightrag_context",
        "constitution",
        "model_name",
        "model_tier",
        "context_window",
        "parent_task_id",
        "project_id",
        "domain",
        "code_context",
        "dependencies",
        "output_format",
    }

    # Pattern to match {variable} placeholders
    VARIABLE_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(self):
        """Initialize the prompt renderer."""
        self._loader = get_prompt_loader()

    def render(
        self,
        prompt: LoadedPrompt | str,
        variables: dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a prompt template with variable substitution.

        Args:
            prompt: LoadedPrompt or raw template string
            variables: Dictionary of variable values
            strict: If True, raise error for missing variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If strict=True and required variables are missing
        """
        content = prompt.content if isinstance(prompt, LoadedPrompt) else prompt

        if strict:
            missing = self._find_missing_variables(content, variables)
            if missing:
                raise ValueError(f"Missing required variables: {missing}")

        # Perform substitution
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                value = variables[var_name]
                # Convert non-strings to string representation
                if isinstance(value, (list, dict)):
                    return self._format_complex_value(value)
                return str(value)
            # Leave unmatched variables as-is
            return match.group(0)

        return self.VARIABLE_PATTERN.sub(replace_var, content)

    def render_agent(
        self,
        agent_name: str,
        variables: dict[str, Any],
        strict: bool = False,
    ) -> tuple[str, str]:
        """
        Load and render an agent prompt.

        Args:
            agent_name: Name of the agent (coordinator, decomposer, etc.)
            variables: Variable values to substitute
            strict: If True, raise error for missing variables

        Returns:
            Tuple of (rendered_content, version_hash)
        """
        prompt = self._loader.get_agent_prompt(agent_name)
        rendered = self.render(prompt, variables, strict)
        return rendered, prompt.version_hash

    def render_worker(
        self,
        worker_name: str,
        variables: dict[str, Any],
        strict: bool = False,
    ) -> tuple[str, str]:
        """
        Load and render a worker prompt.

        Args:
            worker_name: Name of the worker (code_general, documentation, etc.)
            variables: Variable values to substitute
            strict: If True, raise error for missing variables

        Returns:
            Tuple of (rendered_content, version_hash)
        """
        prompt = self._loader.get_worker_prompt(worker_name)
        rendered = self.render(prompt, variables, strict)
        return rendered, prompt.version_hash

    def extract_variables(self, prompt: LoadedPrompt | str) -> set[str]:
        """
        Extract all variable names from a prompt template.

        Args:
            prompt: LoadedPrompt or raw template string

        Returns:
            Set of variable names found in template
        """
        content = prompt.content if isinstance(prompt, LoadedPrompt) else prompt
        return set(self.VARIABLE_PATTERN.findall(content))

    def validate_variables(
        self,
        prompt: LoadedPrompt | str,
        variables: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        """
        Validate variables against a prompt template.

        Args:
            prompt: LoadedPrompt or raw template string
            variables: Provided variables

        Returns:
            Tuple of (missing_variables, extra_variables)
        """
        required = self.extract_variables(prompt)
        provided = set(variables.keys())

        missing = required - provided
        extra = provided - required

        return missing, extra

    def _find_missing_variables(
        self,
        content: str,
        variables: dict[str, Any],
    ) -> set[str]:
        """Find variables in template that aren't provided."""
        required = set(self.VARIABLE_PATTERN.findall(content))
        return required - set(variables.keys())

    @staticmethod
    def _format_complex_value(value: Any) -> str:
        """Format complex values (lists, dicts) for prompt injection."""
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return "\n".join(f"- {item}" for item in value)
            return "\n".join(str(item) for item in value)
        elif isinstance(value, dict):
            lines = []
            for k, v in value.items():
                lines.append(f"- {k}: {v}")
            return "\n".join(lines)
        return str(value)


def render_prompt(
    category: str,
    name: str,
    variables: Optional[dict[str, Any]] = None,
    strict: bool = False,
) -> tuple[str, str]:
    """
    Convenience function to load and render a prompt.

    Args:
        category: Prompt category (\"agents\" or \"workers\")
        name: Prompt name
        variables: Variable values (defaults to empty dict)
        strict: If True, raise error for missing variables

    Returns:
        Tuple of (rendered_content, version_hash)
    """
    renderer = PromptRenderer()
    loader = get_prompt_loader()
    prompt = loader.get(category, name)
    rendered = renderer.render(prompt, variables or {}, strict)
    return rendered, prompt.version_hash