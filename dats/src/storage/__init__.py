"""
Storage module for DATS.

Provides provenance tracking and work product storage abstractions.
"""

from src.storage.provenance import ProvenanceRecord, ProvenanceTracker
from src.storage.work_product import WorkProductStore, Artifact

__all__ = [
    "ProvenanceRecord",
    "ProvenanceTracker",
    "WorkProductStore",
    "Artifact",
]