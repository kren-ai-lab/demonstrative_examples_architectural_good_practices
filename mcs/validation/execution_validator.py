from __future__ import annotations

from typing import List

from mcs.schemas.execution import ExecutionSpec
from mcs.validation.types import Severity, ValidationIssue


def _issue(
    severity: Severity,
    code: str,
    message: str,
    path: str | None = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def validate_execution_spec(spec: ExecutionSpec) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Python version string sanity (lightweight)
    if not spec.python_version.strip():
        issues.append(_issue("ERROR", "EXEC_PYTHON_VERSION_EMPTY", "python_version must be non-empty.", path="python_version"))

    # Git commit recommended
    if spec.git_commit is None:
        issues.append(
            _issue(
                "INFO",
                "EXEC_NO_GIT_COMMIT",
                "git_commit not declared (recommended for auditability).",
                path="git_commit",
                hint="Set git_commit to the repository commit hash for this run.",
            )
        )

    # Dependencies reference required by schema; still validate readability
    if not spec.dependencies.reference.strip():
        issues.append(
            _issue(
                "ERROR",
                "EXEC_DEPS_REF_EMPTY",
                "dependencies.reference must be non-empty.",
                path="dependencies.reference",
            )
        )
    if spec.dependencies.checksum is None:
        issues.append(
            _issue(
                "INFO",
                "EXEC_DEPS_NO_CHECKSUM",
                "dependencies.checksum not declared (recommended).",
                path="dependencies.checksum",
            )
        )

    # Hardware hints
    if spec.hardware:
        if spec.hardware.accelerator == "cuda" and not spec.hardware.gpu:
            issues.append(
                _issue(
                    "WARNING",
                    "EXEC_CUDA_NO_GPU_NAME",
                    "hardware.accelerator='cuda' but hardware.gpu is empty.",
                    path="hardware.gpu",
                )
            )

    # Determinism flags recommended
    if len(spec.determinism_flags) == 0:
        issues.append(
            _issue(
                "INFO",
                "EXEC_NO_DETERMINISM_FLAGS",
                "No determinism_flags declared.",
                path="determinism_flags",
            )
        )

    return issues
