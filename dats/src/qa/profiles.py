"""
QA Profile definitions for DATS.

Defines the different QA validation profiles that can be applied
to worker outputs for quality assurance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QAProfileType(str, Enum):
    """Types of QA profiles available."""

    CONSENSUS = "consensus"
    ADVERSARIAL = "adversarial"
    SECURITY = "security"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    HUMAN = "human"


@dataclass
class QAProfile(ABC):
    """
    Abstract base class for QA profiles.

    Each profile defines how validation should be performed
    for a particular type of quality check.
    """

    profile_type: QAProfileType
    enabled: bool = True
    priority: int = 0  # Higher priority runs first
    config: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for the validator."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile_type": self.profile_type.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "config": self.config,
        }


@dataclass
class ConsensusProfile(QAProfile):
    """
    Consensus-based validation profile.

    Requires multiple reviewers to agree on the validation verdict.
    Disagreements trigger escalation or additional reviewers.
    """

    profile_type: QAProfileType = field(default=QAProfileType.CONSENSUS, init=False)
    min_reviewers: int = 2
    max_reviewers: int = 3
    agreement_threshold: float = 1.0  # 1.0 = all must agree
    prefer_diverse_models: bool = True
    add_reviewer_on_disagreement: bool = True
    escalate_on_persistent_disagreement: bool = True

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for consensus validator."""
        return {
            "min_reviewers": self.min_reviewers,
            "max_reviewers": self.max_reviewers,
            "agreement_threshold": self.agreement_threshold,
            "prefer_diverse_models": self.prefer_diverse_models,
            "add_reviewer_on_disagreement": self.add_reviewer_on_disagreement,
            "escalate_on_persistent_disagreement": self.escalate_on_persistent_disagreement,
            **self.config,
        }


@dataclass
class AdversarialProfile(QAProfile):
    """
    Adversarial review profile.

    Single reviewer in adversarial mode, actively looking for
    flaws, edge cases, and potential failures.
    """

    profile_type: QAProfileType = field(default=QAProfileType.ADVERSARIAL, init=False)
    intensity: str = "high"  # low, medium, high
    max_issues_to_report: int = 10
    focus_areas: list[str] = field(default_factory=list)
    block_on_critical: bool = True
    block_on_major: bool = False

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for adversarial validator."""
        return {
            "intensity": self.intensity,
            "max_issues_to_report": self.max_issues_to_report,
            "focus_areas": self.focus_areas,
            "block_on_critical": self.block_on_critical,
            "block_on_major": self.block_on_major,
            **self.config,
        }


@dataclass
class SecurityProfile(QAProfile):
    """
    Security-focused validation profile.

    Checks for common vulnerabilities, input sanitization,
    error handling, and information leakage.
    """

    profile_type: QAProfileType = field(default=QAProfileType.SECURITY, init=False)
    priority: int = 100  # Security checks run first
    check_injection: bool = True
    check_auth: bool = True
    check_exposure: bool = True
    check_error_handling: bool = True
    check_input_validation: bool = True
    critical_always_blocks: bool = True
    vulnerability_categories: list[str] = field(
        default_factory=lambda: [
            "sql_injection",
            "xss",
            "command_injection",
            "path_traversal",
            "authentication",
            "authorization",
            "sensitive_data_exposure",
            "insecure_deserialization",
        ]
    )

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for security validator."""
        return {
            "check_injection": self.check_injection,
            "check_auth": self.check_auth,
            "check_exposure": self.check_exposure,
            "check_error_handling": self.check_error_handling,
            "check_input_validation": self.check_input_validation,
            "critical_always_blocks": self.critical_always_blocks,
            "vulnerability_categories": self.vulnerability_categories,
            **self.config,
        }


@dataclass
class TestingProfile(QAProfile):
    """
    Testing validation profile.

    Validates test coverage, assertion quality, and edge case testing.
    """

    profile_type: QAProfileType = field(default=QAProfileType.TESTING, init=False)
    check_coverage: bool = True
    check_assertion_quality: bool = True
    check_edge_cases: bool = True
    min_coverage_threshold: float = 0.8
    can_request_additional_tests: bool = True
    fail_on_missing_tests: bool = False

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for testing validator."""
        return {
            "check_coverage": self.check_coverage,
            "check_assertion_quality": self.check_assertion_quality,
            "check_edge_cases": self.check_edge_cases,
            "min_coverage_threshold": self.min_coverage_threshold,
            "can_request_additional_tests": self.can_request_additional_tests,
            "fail_on_missing_tests": self.fail_on_missing_tests,
            **self.config,
        }


@dataclass
class DocumentationProfile(QAProfile):
    """
    Documentation accuracy validation profile.

    Cross-references documentation against source material,
    uses Q&A validation, and checks completeness.
    """

    profile_type: QAProfileType = field(default=QAProfileType.DOCUMENTATION, init=False)
    cross_reference_sources: bool = True
    use_qa_validation: bool = True
    check_completeness: bool = True
    flag_uncertainty: bool = True
    source_materials: list[str] = field(default_factory=list)
    scope_definition: Optional[str] = None

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for documentation validator."""
        return {
            "cross_reference_sources": self.cross_reference_sources,
            "use_qa_validation": self.use_qa_validation,
            "check_completeness": self.check_completeness,
            "flag_uncertainty": self.flag_uncertainty,
            "source_materials": self.source_materials,
            "scope_definition": self.scope_definition,
            **self.config,
        }


@dataclass
class HumanProfile(QAProfile):
    """
    Human review profile.

    Queues work for human review with timeout and escalation handling.
    """

    profile_type: QAProfileType = field(default=QAProfileType.HUMAN, init=False)
    timeout_hours: int = 24
    reminder_hours: int = 4
    escalation_on_timeout: bool = True
    priority_level: str = "normal"  # low, normal, high, critical
    required_reviewers: int = 1
    specific_reviewers: list[str] = field(default_factory=list)
    decision_points: list[str] = field(default_factory=list)

    def get_validator_config(self) -> dict[str, Any]:
        """Get configuration for human validator."""
        return {
            "timeout_hours": self.timeout_hours,
            "reminder_hours": self.reminder_hours,
            "escalation_on_timeout": self.escalation_on_timeout,
            "priority_level": self.priority_level,
            "required_reviewers": self.required_reviewers,
            "specific_reviewers": self.specific_reviewers,
            "decision_points": self.decision_points,
            **self.config,
        }


# Default profile configurations
DEFAULT_PROFILES: dict[QAProfileType, QAProfile] = {
    QAProfileType.CONSENSUS: ConsensusProfile(),
    QAProfileType.ADVERSARIAL: AdversarialProfile(),
    QAProfileType.SECURITY: SecurityProfile(),
    QAProfileType.TESTING: TestingProfile(),
    QAProfileType.DOCUMENTATION: DocumentationProfile(),
    QAProfileType.HUMAN: HumanProfile(),
}


def get_profile(
    profile_type: QAProfileType | str,
    **overrides: Any,
) -> QAProfile:
    """
    Get a QA profile by type with optional overrides.

    Args:
        profile_type: Profile type to get
        **overrides: Configuration overrides

    Returns:
        Configured QAProfile instance
    """
    if isinstance(profile_type, str):
        profile_type = QAProfileType(profile_type)

    # Get the default profile class
    profile_classes = {
        QAProfileType.CONSENSUS: ConsensusProfile,
        QAProfileType.ADVERSARIAL: AdversarialProfile,
        QAProfileType.SECURITY: SecurityProfile,
        QAProfileType.TESTING: TestingProfile,
        QAProfileType.DOCUMENTATION: DocumentationProfile,
        QAProfileType.HUMAN: HumanProfile,
    }

    profile_class = profile_classes.get(profile_type)
    if not profile_class:
        raise ValueError(f"Unknown profile type: {profile_type}")

    # Create profile with overrides
    return profile_class(**overrides)


@dataclass
class QAProfileSet:
    """
    A set of QA profiles to apply to a validation.

    Manages multiple profiles and their execution order.
    """

    primary_profile: QAProfile
    additional_checks: list[QAProfile] = field(default_factory=list)

    def get_all_profiles(self) -> list[QAProfile]:
        """Get all profiles in execution order (by priority)."""
        all_profiles = [self.primary_profile] + self.additional_checks
        # Sort by priority (higher first), then by type for consistency
        return sorted(
            all_profiles,
            key=lambda p: (-p.priority, p.profile_type.value),
        )

    def has_profile(self, profile_type: QAProfileType) -> bool:
        """Check if a profile type is included."""
        if self.primary_profile.profile_type == profile_type:
            return True
        return any(p.profile_type == profile_type for p in self.additional_checks)

    def get_profile(self, profile_type: QAProfileType) -> Optional[QAProfile]:
        """Get a specific profile if included."""
        if self.primary_profile.profile_type == profile_type:
            return self.primary_profile
        for p in self.additional_checks:
            if p.profile_type == profile_type:
                return p
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QAProfileSet":
        """
        Create a profile set from configuration.

        Args:
            config: Configuration dict with 'primary' and 'additional' keys

        Returns:
            Configured QAProfileSet
        """
        primary_config = config.get("primary", {})
        primary_type = primary_config.get("type", "consensus")
        primary_overrides = {k: v for k, v in primary_config.items() if k != "type"}
        primary = get_profile(primary_type, **primary_overrides)

        additional = []
        for check_config in config.get("additional", []):
            check_type = check_config.get("type")
            if check_type:
                check_overrides = {k: v for k, v in check_config.items() if k != "type"}
                additional.append(get_profile(check_type, **check_overrides))

        return cls(primary_profile=primary, additional_checks=additional)