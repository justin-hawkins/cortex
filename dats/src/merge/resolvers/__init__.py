"""
Conflict resolvers for merge coordination.

Provides different resolution strategies for different conflict types.
"""

from src.merge.resolvers.base import (
    BaseResolver,
    MergeEngine,
    MergeResult,
    ResolverResult,
)
from src.merge.resolvers.textual import TextualResolver, DifflibMergeEngine
from src.merge.resolvers.semantic import SemanticResolver
from src.merge.resolvers.architectural import ArchitecturalResolver

__all__ = [
    # Base
    "BaseResolver",
    "MergeEngine",
    "MergeResult",
    "ResolverResult",
    # Implementations
    "TextualResolver",
    "DifflibMergeEngine",
    "SemanticResolver",
    "ArchitecturalResolver",
]