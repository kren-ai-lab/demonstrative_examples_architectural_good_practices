from __future__ import annotations

from typing import List

from mcs.schemas.split import SplitSpec
from mcs.validation.types import Severity, ValidationIssue


def _issue(
    severity: Severity,
    code: str,
    message: str,
    path: str | None = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def validate_split_spec(spec: SplitSpec) -> List[ValidationIssue]:
    """
    Semantic validator for SplitSpec (beyond Pydantic).
    Enforces auditability rather than prescribing a specific splitter.
    """
    issues: List[ValidationIssue] = []

    # Seed sanity
    if spec.seed is None:
        issues.append(
            _issue("ERROR", "SPLIT_NO_SEED", "SplitSpec.seed must be set for determinism.", path="seed")
        )
    elif spec.seed < 0:
        issues.append(
            _issue("WARNING", "SPLIT_SEED_NEGATIVE", "Seed is negative; allowed but uncommon.", path="seed")
        )

    # Leakage checks policy
    if spec.protocol == "random" and len(spec.leakage_checks) == 0:
        issues.append(
            _issue(
                "INFO",
                "SPLIT_RANDOM_NO_CHECKS",
                "Random split has no leakage checks declared.",
                path="leakage_checks",
                hint="Consider at least exact_duplicate across 'all'.",
            )
        )

    if spec.protocol != "random" and len(spec.leakage_checks) == 0:
        issues.append(
            _issue(
                "ERROR",
                "SPLIT_NO_CHECKS_NONRANDOM",
                "Non-random split protocols must declare leakage_checks for auditability.",
                path="leakage_checks",
            )
        )

    # Group protocol -> recommend group_overlap
    if spec.protocol == "group" and spec.group_field:
        if not any(c.type == "group_overlap" for c in spec.leakage_checks):
            issues.append(
                _issue(
                    "WARNING",
                    "SPLIT_GROUP_NO_GROUP_OVERLAP_CHECK",
                    "Group split declared but no group_overlap leakage check found.",
                    path="leakage_checks",
                    hint="Add LeakageCheck(type='group_overlap', field=group_field).",
                )
            )

    # Identity-aware
    if spec.protocol in {"cluster_aware", "structure_aware"}:
        if spec.identity is None:
            issues.append(
                _issue(
                    "ERROR",
                    "SPLIT_IDENTITY_MISSING",
                    f"identity must be declared when protocol='{spec.protocol}'.",
                    path="identity",
                )
            )
        else:
            if spec.identity.measure in {"sequence_identity", "embedding_similarity"}:
                if spec.identity.threshold is None:
                    issues.append(
                        _issue(
                            "ERROR",
                            "SPLIT_IDENTITY_THRESHOLD_MISSING",
                            "identity.threshold is required for similarity-based measures.",
                            path="identity.threshold",
                        )
                    )
                elif spec.identity.threshold >= 0.8:
                    issues.append(
                        _issue(
                            "WARNING",
                            "SPLIT_IDENTITY_THRESHOLD_HIGH",
                            f"identity.threshold={spec.identity.threshold} is very high; may allow close homologs across partitions.",
                            path="identity.threshold",
                        )
                    )

    # Temporal
    if spec.protocol == "temporal" and len(spec.leakage_checks) == 0:
        issues.append(
            _issue(
                "WARNING",
                "SPLIT_TEMPORAL_NO_CHECKS",
                "Temporal split has no leakage checks; consider exact_duplicate (and group_overlap if relevant).",
                path="leakage_checks",
            )
        )

    # Encourage exact duplicates check always
    if not any(c.type == "exact_duplicate" for c in spec.leakage_checks):
        issues.append(
            _issue(
                "INFO",
                "SPLIT_NO_EXACT_DUP_CHECK",
                "No exact_duplicate leakage check declared.",
                path="leakage_checks",
                hint="Add exact_duplicate across 'all'.",
            )
        )

    # Ratios
    if spec.ratios.val == 0.0:
        issues.append(
            _issue(
                "INFO",
                "SPLIT_NO_VALIDATION_SET",
                "Validation ratio is 0.0; ensure training/evaluation design reflects this (e.g., CV).",
                path="ratios.val",
            )
        )

    return issues
