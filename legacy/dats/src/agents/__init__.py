"""
Agent module for DATS.

Provides orchestration agents that coordinate task decomposition,
execution, quality assurance, and result merging.
"""

from src.agents.base import BaseAgent
from src.agents.coordinator import Coordinator
from src.agents.decomposer import Decomposer
from src.agents.complexity_estimator import ComplexityEstimator
from src.agents.qa_reviewer import QAReviewer
from src.agents.merge_coordinator import MergeCoordinator

__all__ = [
    "BaseAgent",
    "Coordinator",
    "Decomposer",
    "ComplexityEstimator",
    "QAReviewer",
    "MergeCoordinator",
]