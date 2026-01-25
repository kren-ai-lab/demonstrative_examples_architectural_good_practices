from __future__ import annotations

from typing import List, Optional

from mcs.schemas.dataset import DatasetSpec
from mcs.schemas.split import SplitSpec
from mcs.schemas.embedding import EmbeddingSpec
from mcs.schemas.train import TrainSpec
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


# ---------------------------------------------------------------------
# v0 checks (dataset ↔ split) — keep the public function as-is
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Unified extended checks (dataset ↔ split ↔ embedding ↔ train ↔ execution)
# ---------------------------------------------------------------------

def validate_consistency(
    dataset: DatasetSpec,
    split: SplitSpec,
    embedding: Optional[EmbeddingSpec] = None,
    train: Optional[TrainSpec] = None,
    execution: Optional[ExecutionSpec] = None,
) -> List[ValidationIssue]:
    """
    Unified cross-spec consistency checks.

    Notes
    -----
    - Always runs dataset↔split checks.
    - Additional checks run only if the corresponding spec is provided.
    """
    issues: List[ValidationIssue] = []

    # Always include the original v0 checks
    issues.extend(validate_dataset_split_consistency(dataset, split))

    # -------------------------
    # dataset ↔ embedding
    # -------------------------
    if embedding is not None:
        # If embedding declares dataset fingerprint, it should match
        if embedding.input_dataset_fingerprint is not None:
            ds_fp = dataset.fingerprint()
            if embedding.input_dataset_fingerprint != ds_fp:
                issues.append(
                    _issue(
                        "WARNING",
                        "CONSIST_EMB_DATASET_FP_MISMATCH",
                        "EmbeddingSpec.input_dataset_fingerprint does not match DatasetSpec.fingerprint().",
                        path="embedding.input_dataset_fingerprint",
                        hint="Update input_dataset_fingerprint after computing DatasetSpec.fingerprint().",
                    )
                )

        # Embedding output declared but dataset has no sequence_field hygiene? (very light)
        if not dataset.sequence_field or not dataset.sequence_field.strip():
            issues.append(
                _issue(
                    "ERROR",
                    "CONSIST_DATASET_NO_SEQUENCE_FIELD",
                    "DatasetSpec.sequence_field must be a non-empty string for embedding extraction.",
                    path="dataset.sequence_field",
                )
            )

    # -------------------------
    # train ↔ dataset / split / embedding
    # -------------------------
    if train is not None:
        # Supervised task requires labels
        if train.task != "unsupervised" and len(dataset.labels) == 0:
            issues.append(
                _issue(
                    "ERROR",
                    "CONSIST_TRAIN_SUPERVISED_NO_LABELS",
                    f"TrainSpec.task='{train.task}' but DatasetSpec.labels is empty.",
                    path="dataset.labels",
                    hint="Add label fields to DatasetSpec or set task='unsupervised'.",
                )
            )

        # If train declares linkage fingerprints, they should match
        if train.input_split_fingerprint is not None:
            sp_fp = split.fingerprint()
            if train.input_split_fingerprint != sp_fp:
                issues.append(
                    _issue(
                        "WARNING",
                        "CONSIST_TRAIN_SPLIT_FP_MISMATCH",
                        "TrainSpec.input_split_fingerprint does not match SplitSpec.fingerprint().",
                        path="train.input_split_fingerprint",
                        hint="Update input_split_fingerprint after computing SplitSpec.fingerprint().",
                    )
                )
        else:
            issues.append(
                _issue(
                    "INFO",
                    "TRAIN_NO_SPLIT_LINK",
                    "TrainSpec.input_split_fingerprint not set (optional).",
                    path="train.input_split_fingerprint",
                    hint="Fill after computing SplitSpec.fingerprint().",
                )
            )

        if embedding is not None:
            if train.input_embedding_fingerprint is not None:
                em_fp = embedding.fingerprint()
                if train.input_embedding_fingerprint != em_fp:
                    issues.append(
                        _issue(
                            "WARNING",
                            "CONSIST_TRAIN_EMB_FP_MISMATCH",
                            "TrainSpec.input_embedding_fingerprint does not match EmbeddingSpec.fingerprint().",
                            path="train.input_embedding_fingerprint",
                            hint="Update input_embedding_fingerprint after computing EmbeddingSpec.fingerprint().",
                        )
                    )
            else:
                issues.append(
                    _issue(
                        "INFO",
                        "TRAIN_NO_EMB_LINK",
                        "TrainSpec.input_embedding_fingerprint not set (optional).",
                        path="train.input_embedding_fingerprint",
                        hint="Fill after computing EmbeddingSpec.fingerprint().",
                    )
                )

        # If no validation partition, recommend CV (info-level)
        if getattr(split, "ratios", None) is not None:
            if split.ratios.val == 0.0 and (train.cross_validation is None or not train.cross_validation.enabled):
                issues.append(
                    _issue(
                        "INFO",
                        "CONSIST_NO_VAL_RECOMMEND_CV",
                        "Validation ratio is 0.0 and CV is disabled; consider enabling CV or adding a validation partition.",
                        path="train.cross_validation",
                    )
                )
            if split.ratios.val == 0.0 and split.ratios.test == 0.0:
                issues.append(
                    _issue(
                        "WARNING",
                        "CONSIST_NO_VAL_NO_TEST",
                        "Split has neither validation nor test partition; ensure evaluation design is explicit.",
                        path="split.ratios",
                    )
                )

    # -------------------------
    # execution ↔ all (auditability reminders)
    # -------------------------
    if execution is not None:
        if execution.git_commit is None:
            issues.append(
                _issue(
                    "INFO",
                    "CONSIST_NO_GIT_COMMIT",
                    "ExecutionSpec.git_commit missing; reduces auditability across runs.",
                    path="execution.git_commit",
                )
            )

        # If execution says cuda but embedding says cpu (or vice versa), that's not necessarily wrong,
        # but it can be surprising—flag as INFO.
        if embedding is not None and execution.hardware is not None:
            if execution.hardware.accelerator == "cuda" and embedding.device == "cpu":
                issues.append(
                    _issue(
                        "INFO",
                        "CONSIST_CUDA_EXEC_CPU_EMB",
                        "Execution uses CUDA but EmbeddingSpec.device='cpu'. This is valid if embedding extraction is CPU-bound, but verify intent.",
                        path="embedding.device",
                    )
                )
            if execution.hardware.accelerator in {"none", "mps", "tpu", "other"} and embedding.device == "cuda":
                issues.append(
                    _issue(
                        "INFO",
                        "CONSIST_NONCUDA_EXEC_CUDA_EMB",
                        "EmbeddingSpec.device='cuda' but ExecutionSpec.hardware.accelerator is not 'cuda'. Verify consistency.",
                        path="execution.hardware.accelerator",
                    )
                )

    return issues
