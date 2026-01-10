"""
Prompt template loader for DATS.

Loads and caches prompt templates from the prompts directory.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings


@dataclass
class LoadedPrompt:
    """A loaded prompt template with metadata."""

    content: str
    version_hash: str
    category: str
    name: str
    path: Path

    def __str__(self) -> str:
        return self.content


class PromptLoader:
    """
    Load and cache prompt templates.

    Prompts are organized by category (agents, workers) and loaded
    from markdown files. Each prompt gets a version hash for provenance.
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Path to prompts directory. Defaults to settings value.
        """
        if prompts_dir is None:
            prompts_dir = get_settings().prompts_dir
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[tuple[str, str], LoadedPrompt] = {}

    def get(
        self,
        category: str,
        name: str,
        version: str = "latest",
    ) -> LoadedPrompt:
        """
        Load a prompt template.

        Args:
            category: Prompt category ("agents" or "workers")
            name: Prompt name (without extension)
            version: Version string (currently only "latest" supported)

        Returns:
            LoadedPrompt with content and metadata

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        cache_key = (category, name)

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build path and load
        path = self.prompts_dir / category / f"{name}.md"

        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        content = path.read_text()
        version_hash = self._compute_hash(content)

        prompt = LoadedPrompt(
            content=content,
            version_hash=version_hash,
            category=category,
            name=name,
            path=path,
        )

        # Cache it
        self._cache[cache_key] = prompt

        return prompt

    def get_agent_prompt(self, name: str) -> LoadedPrompt:
        """
        Load an agent prompt template.

        Args:
            name: Agent name (coordinator, decomposer, etc.)

        Returns:
            LoadedPrompt
        """
        return self.get("agents", name)

    def get_worker_prompt(self, name: str) -> LoadedPrompt:
        """
        Load a worker prompt template.

        Args:
            name: Worker name (code_general, documentation, etc.)

        Returns:
            LoadedPrompt
        """
        return self.get("workers", name)

    def list_prompts(self, category: Optional[str] = None) -> list[tuple[str, str]]:
        """
        List available prompts.

        Args:
            category: Optional category filter

        Returns:
            List of (category, name) tuples
        """
        prompts = []
        categories = [category] if category else ["agents", "workers"]

        for cat in categories:
            cat_dir = self.prompts_dir / cat
            if cat_dir.exists():
                for path in cat_dir.glob("*.md"):
                    prompts.append((cat, path.stem))

        return prompts

    def reload(self, category: Optional[str] = None, name: Optional[str] = None):
        """
        Clear cache to force reload.

        Args:
            category: Optional category to clear
            name: Optional name to clear (requires category)
        """
        if category and name:
            key = (category, name)
            self._cache.pop(key, None)
        elif category:
            keys_to_remove = [k for k in self._cache if k[0] == category]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

    @staticmethod
    def _compute_hash(content: str) -> str:
        """
        Compute version hash for content.

        Args:
            content: Prompt content

        Returns:
            12-character SHA256 hash
        """
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# Cached singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_dir: Optional[str] = None) -> PromptLoader:
    """
    Get cached prompt loader instance.

    Args:
        prompts_dir: Optional prompts directory path

    Returns:
        Cached PromptLoader instance
    """
    global _prompt_loader

    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_dir)

    return _prompt_loader