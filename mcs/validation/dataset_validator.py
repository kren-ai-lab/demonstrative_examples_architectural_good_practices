from __future__ import annotations

from typing import List

from mcs.schemas.dataset import DatasetSpec, FilterRule, RedundancyControl
from mcs.validation.types import Severity, ValidationIssue


def _issue(
    severity: Severity,
    code: str,
    message: str,
    path: str | None = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def validate_dataset_spec(spec: DatasetSpec) -> List[ValidationIssue]:
    """
    Semantic validator for DatasetSpec (beyond Pydantic structural validation).
    """
    issues: List[ValidationIssue] = []

    # Provenance strength
    if not spec.sources:
        issues.append(
            _issue(
                "ERROR",
                "DATASET_NO_SOURCES",
                "DatasetSpec must declare at least one provenance source.",
                path="sources",
                hint="Add at least one DatasetSource entry.",
            )
        )

    for i, s in enumerate(spec.sources):
        if s.type in {"database", "url"} and not (s.version or s.identifier):
            issues.append(
                _issue(
                    "WARNING",
                    "DATASET_SOURCE_UNVERSIONED",
                    f"Source '{s.name}' is mutable but has no explicit version/identifier.",
                    path=f"sources[{i}]",
                    hint="Provide 'version' (release/snapshot date) or 'identifier' (DOI/commit/Zenodo record).",
                )
            )

    # Labels present?
    if len(spec.labels) == 0:
        issues.append(
            _issue(
                "INFO",
                "DATASET_UNLABELED",
                "DatasetSpec declares no labels. This is fine for unsupervised workflows.",
                path="labels",
            )
        )

    # Required metadata without description
    for i, mf in enumerate(spec.metadata):
        if mf.required and not mf.description:
            issues.append(
                _issue(
                    "WARNING",
                    "DATASET_REQUIRED_META_NO_DESC",
                    f"Metadata field '{mf.name}' is required but has no description.",
                    path=f"metadata[{i}]",
                    hint="Add a short description for portability.",
                )
            )

    # Filters sanity
    for i, fr in enumerate(spec.filters):
        issues.extend(_validate_filter_rule(fr, index=i))

    # Redundancy sanity
    issues.extend(_validate_redundancy(spec.redundancy_control))

    # Output hints
    if spec.output is None:
        issues.append(
            _issue(
                "INFO",
                "DATASET_NO_OUTPUT",
                "DatasetSpec has no output materialisation info. Acceptable, but reduces traceability.",
                path="output",
                hint="Consider adding output.format and output.path for registry integration.",
            )
        )
    else:
        if not spec.output.path.strip():
            issues.append(
                _issue(
                    "ERROR",
                    "DATASET_OUTPUT_EMPTY_PATH",
                    "Dataset output.path cannot be empty.",
                    path="output.path",
                )
            )
        if spec.output.checksum is None:
            issues.append(
                _issue(
                    "INFO",
                    "DATASET_OUTPUT_NO_CHECKSUM",
                    "Dataset output has no checksum declared.",
                    path="output.checksum",
                    hint="Optional but recommended for file-based reproducibility.",
                )
            )

    return issues


def _validate_filter_rule(rule: FilterRule, index: int) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    p = f"filters[{index}]"

    if rule.op == "between":
        if not (isinstance(rule.value, (list, tuple)) and len(rule.value) == 2):
            issues.append(
                _issue(
                    "ERROR",
                    "FILTER_BETWEEN_SHAPE",
                    "Filter op='between' requires value=[low, high].",
                    path=f"{p}.value",
                )
            )
        else:
            low, high = rule.value
            if low is None or high is None:
                issues.append(
                    _issue(
                        "ERROR",
                        "FILTER_BETWEEN_NONE",
                        "Filter op='between' cannot have None bounds.",
                        path=f"{p}.value",
                    )
                )

    if rule.op in {"in", "not_in"}:
        if not isinstance(rule.value, (list, tuple)):
            issues.append(
                _issue(
                    "ERROR",
                    "FILTER_IN_TYPE",
                    f"Filter op='{rule.op}' requires a list/tuple value.",
                    path=f"{p}.value",
                )
            )
        elif len(rule.value) == 0:
            issues.append(
                _issue(
                    "WARNING",
                    "FILTER_IN_EMPTY",
                    f"Filter op='{rule.op}' has an empty list; semantics may be degenerate.",
                    path=f"{p}.value",
                )
            )

    if rule.op == "regex":
        if not isinstance(rule.value, str) or not rule.value.strip():
            issues.append(
                _issue(
                    "ERROR",
                    "FILTER_REGEX_EMPTY",
                    "Filter op='regex' requires a non-empty regex string.",
                    path=f"{p}.value",
                )
            )

    if rule.op == "exists":
        if rule.value not in (None, True, False):
            issues.append(
                _issue(
                    "WARNING",
                    "FILTER_EXISTS_VALUE",
                    "Filter op='exists' typically uses value true/false (or null).",
                    path=f"{p}.value",
                )
            )

    if not rule.description:
        issues.append(
            _issue(
                "INFO",
                "FILTER_NO_DESC",
                f"Filter rule '{rule.field}:{rule.op}' has no description.",
                path=f"{p}.description",
            )
        )

    return issues


def _validate_redundancy(rc: RedundancyControl) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if rc.method == "none":
        issues.append(
            _issue(
                "INFO",
                "REDUNDANCY_NONE",
                "No redundancy control declared. This may inflate effective sample size.",
                path="redundancy_control.method",
            )
        )

    if rc.method in {"sequence_identity", "cluster"} and rc.tool is None:
        issues.append(
            _issue(
                "INFO",
                "REDUNDANCY_NO_TOOL",
                f"Redundancy method '{rc.method}' has no tool declared.",
                path="redundancy_control.tool",
                hint="Optional but recommended (e.g., mmseqs2, cd-hit).",
            )
        )

    if rc.method == "custom" and not (rc.tool or rc.parameters):
        issues.append(
            _issue(
                "WARNING",
                "REDUNDANCY_CUSTOM_UNSPECIFIED",
                "Custom redundancy control declared but no tool/parameters provided.",
                path="redundancy_control",
            )
        )

    return issues
