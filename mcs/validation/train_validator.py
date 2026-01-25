from __future__ import annotations

from typing import List

from mcs.schemas.train import TrainSpec
from mcs.validation.types import Severity, ValidationIssue


def _issue(
    severity: Severity,
    code: str,
    message: str,
    path: str | None = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def validate_train_spec(spec: TrainSpec) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Seed
    if spec.seed is None:
        issues.append(_issue("ERROR", "TRAIN_NO_SEED", "TrainSpec.seed must be set for determinism.", path="seed"))
    elif spec.seed < 0:
        issues.append(_issue("WARNING", "TRAIN_NEGATIVE_SEED", "Negative seed is allowed but uncommon.", path="seed"))

    # Metrics (schema already enforces for supervised, keep INFO for unsupervised)
    if spec.task == "unsupervised" and len(spec.metrics) > 0:
        issues.append(
            _issue(
                "INFO",
                "TRAIN_UNSUP_WITH_METRICS",
                "Unsupervised task declares metrics; ensure they are appropriate (e.g., silhouette).",
                path="metrics",
            )
        )

    # Hyperparams serializability hints
    if spec.hyperparams and "random_state" not in spec.hyperparams and spec.seed is not None:
        issues.append(
            _issue(
                "INFO",
                "TRAIN_NO_RANDOM_STATE_IN_HYPERPARAMS",
                "hyperparams has no 'random_state' key; ensure estimator uses TrainSpec.seed where applicable.",
                path="hyperparams",
            )
        )

    # CV hints
    if spec.cross_validation and spec.cross_validation.enabled:
        if spec.cross_validation.folds < 2:
            issues.append(
                _issue(
                    "ERROR",
                    "TRAIN_CV_FOLDS_INVALID",
                    "cross_validation.folds must be >= 2 when enabled.",
                    path="cross_validation.folds",
                )
            )

    # Outputs
    if spec.output is None:
        issues.append(
            _issue(
                "INFO",
                "TRAIN_NO_OUTPUT",
                "TrainSpec has no output declared; acceptable but reduces traceability.",
                path="output",
                hint="Consider declaring output paths for model/metrics/predictions.",
            )
        )
    else:
        if spec.output.metrics_path is None:
            issues.append(
                _issue(
                    "INFO",
                    "TRAIN_OUTPUT_NO_METRICS_PATH",
                    "output.metrics_path not declared.",
                    path="output.metrics_path",
                )
            )

    return issues
