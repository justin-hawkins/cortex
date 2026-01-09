"""
Cascade failure handling for DATS.

Provides taint propagation, lazy revalidation, and rollback capabilities
for managing failures that affect downstream task outputs.
"""

from src.cascade.detector import CascadeDetector, CascadeScenario, CascadeMetrics
from src.cascade.taint import TaintPropagator, PropagationResult
from src.cascade.revalidation import (
    RevalidationTask,
    RevalidationQueue,
    RevalidationResult,
    RevalidationVerdict,
    RevalidationEvaluator,
)
from src.cascade.rollback import (
    RollbackManager,
    RollbackResult,
    RollbackTrigger,
)

__all__ = [
    # Detector
    "CascadeDetector",
    "CascadeScenario",
    "CascadeMetrics",
    # Taint
    "TaintPropagator",
    "PropagationResult",
    # Revalidation
    "RevalidationTask",
    "RevalidationQueue",
    "RevalidationResult",
    "RevalidationVerdict",
    "RevalidationEvaluator",
    # Rollback
    "RollbackManager",
    "RollbackResult",
    "RollbackTrigger",
]