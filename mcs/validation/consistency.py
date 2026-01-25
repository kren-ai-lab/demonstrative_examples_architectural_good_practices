from __future__ import annotations

from typing import List

from mcs.schemas.dataset import DatasetSpec
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


def validate_dataset_split_consistency(dataset: DatasetSpec, split: SplitSpec) -> List[ValidationIssue]:
    """
    Cross-spec consistency checks (dataset ↔ split).
    """
    issues: List[ValidationIssue] = []

    label_fields = {l.name for l in dataset.labels}
    meta_fields = {m.name for m in dataset.metadata}
    reserved = {dataset.id_field, dataset.sequence_field}
    known_fields = label_fields | meta_fields | reserved

    # stratify field must exist
    if split.protocol == "stratified":
        f = split.stratify_field
        if f and f not in known_fields:
            issues.append(
                _issue(
                    "ERROR",
                    "CONSIST_STRATIFY_UNKNOWN_FIELD",
                    f"stratify_field='{f}' not declared in DatasetSpec labels/metadata.",
                    path="split.stratify_field",
                    hint="Declare it in DatasetSpec.labels or DatasetSpec.metadata.",
                )
            )

    # group field must exist
    if split.protocol == "group":
        f = split.group_field
        if f and f not in known_fields:
            issues.append(
                _issue(
                    "ERROR",
                    "CONSIST_GROUP_UNKNOWN_FIELD",
                    f"group_field='{f}' not declared in DatasetSpec metadata.",
                    path="split.group_field",
                    hint="Declare it in DatasetSpec.metadata.",
                )
            )

    # temporal field must exist
    if split.protocol == "temporal":
        f = split.time_field
        if f and f not in known_fields:
            issues.append(
                _issue(
                    "ERROR",
                    "CONSIST_TIME_UNKNOWN_FIELD",
                    f"time_field='{f}' not declared in DatasetSpec metadata.",
                    path="split.time_field",
                    hint="Declare it in DatasetSpec.metadata.",
                )
            )

    # group_overlap leakage check field must exist
    for i, chk in enumerate(split.leakage_checks):
        if chk.type == "group_overlap":
            if not chk.field:
                issues.append(
                    _issue(
                        "ERROR",
                        "CONSIST_GROUP_OVERLAP_NO_FIELD",
                        "LeakageCheck(type='group_overlap') requires a field.",
                        path=f"split.leakage_checks[{i}].field",
                    )
                )
            elif chk.field not in known_fields:
                issues.append(
                    _issue(
                        "ERROR",
                        "CONSIST_GROUP_OVERLAP_UNKNOWN_FIELD",
                        f"group_overlap field='{chk.field}' not declared in DatasetSpec.",
                        path=f"split.leakage_checks[{i}].field",
                        hint="Declare it in DatasetSpec.metadata.",
                    )
                )

    # Coherence: identity-aware split but dataset has no redundancy policy (warn)
    if dataset.redundancy_control.method == "none" and split.protocol in {"cluster_aware", "structure_aware"}:
        issues.append(
            _issue(
                "WARNING",
                "CONSIST_NO_REDUNDANCY_WITH_IDENTITY_SPLIT",
                "Split is identity-aware but dataset declares no redundancy control. Valid, but ensure identity computation is clearly declared.",
                path="dataset.redundancy_control.method",
                hint="Consider declaring redundancy_control or specifying identity tool/parameters in SplitSpec.identity.",
            )
        )

    # Coherence: split threshold should typically be stricter than redundancy threshold
    if split.identity and split.identity.measure == "sequence_identity":
        if dataset.redundancy_control.method == "sequence_identity" and dataset.redundancy_control.threshold is not None:
            if split.identity.threshold is not None and split.identity.threshold >= dataset.redundancy_control.threshold:
                issues.append(
                    _issue(
                        "WARNING",
                        "CONSIST_SPLIT_THRESHOLD_GE_REDUNDANCY",
                        f"Split identity threshold ({split.identity.threshold}) >= dataset redundancy threshold ({dataset.redundancy_control.threshold}).",
                        path="split.identity.threshold",
                        hint="Typically, split threshold should be lower (stricter) than redundancy threshold.",
                    )
                )

    # Informational: random split with labels can cause imbalance
    if dataset.labels and split.protocol == "random":
        issues.append(
            _issue(
                "INFO",
                "CONSIST_RANDOM_WITH_LABELS",
                "Dataset has labels but split is random (not stratified). May cause imbalance.",
                path="split.protocol",
                hint="Consider protocol='stratified' if label imbalance matters.",
            )
        )

    return issues
