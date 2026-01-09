"""
QA and Validation Pipeline for DATS.

This module provides comprehensive quality assurance for worker outputs
through multiple validation profiles, consensus handling, and human review.
"""

from src.qa.results import (
    QAVerdict,
    IssueSeverity,
    IssueCategory,
    ProfileVerdict,
    QAIssue,
    ProfileResult,
    RevisionGuidance,
    QAMetadata,
    QAResult,
)
from src.qa.profiles import (
    QAProfileType,
    QAProfile,
    QAProfileSet,
    ConsensusProfile,
    AdversarialProfile,
    SecurityProfile,
    TestingProfile,
    DocumentationProfile,
    HumanProfile,
    get_profile,
)
from src.qa.pipeline import QAPipeline, QAPipelineConfig
from src.qa.human_review import (
    HumanReviewStatus,
    HumanReviewPriority,
    HumanReviewRequest,
    HumanReviewResponse,
    HumanReviewQueue,
)
from src.qa.validators.base import WorkerOutput, ValidatorContext

__all__ = [
    # Results
    "QAVerdict",
    "IssueSeverity",
    "IssueCategory",
    "ProfileVerdict",
    "QAIssue",
    "ProfileResult",
    "RevisionGuidance",
    "QAMetadata",
    "QAResult",
    # Profiles
    "QAProfileType",
    "QAProfile",
    "QAProfileSet",
    "ConsensusProfile",
    "AdversarialProfile",
    "SecurityProfile",
    "TestingProfile",
    "DocumentationProfile",
    "HumanProfile",
    "get_profile",
    # Pipeline
    "QAPipeline",
    "QAPipelineConfig",
    # Human Review
    "HumanReviewStatus",
    "HumanReviewPriority",
    "HumanReviewRequest",
    "HumanReviewResponse",
    "HumanReviewQueue",
    # Validators
    "WorkerOutput",
    "ValidatorContext",
]
