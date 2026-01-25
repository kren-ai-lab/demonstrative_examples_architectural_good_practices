from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Literal
from pathlib import Path
import yaml

from mcs.schemas.dataset import DatasetSpec
from mcs.schemas.split import SplitSpec
from mcs.schemas.embedding import EmbeddingSpec
from mcs.schemas.train import TrainSpec
from mcs.schemas.execution import ExecutionSpec

from mcs.validation.api import (
    format_issues,
    sort_issues,
)
from mcs.validation.dataset_validator import validate_dataset_spec
from mcs.validation.split_validator import validate_split_spec
from mcs.validation.consistency import validate_dataset_split_consistency

from mcs.provenance import ProvenanceRecord, ArtifactRef, MetricSummary
from mcs.registry import LocalRegistry


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at '{p}' must contain a mapping (dict-like) at the top level.")
    return data


def load_specs(
    directory: str | Path,
    *,
    dataset: str = "dataset.yaml",
    split: str = "split.yaml",
    embedding: str = "embedding.yaml",
    train: str = "train.yaml",
    execution: str = "execution.yaml",
) -> Dict[str, Any]:
    """
    Load MCS specs from a directory.

    Notes
    -----
    You can point train/execution to task-specific files like:
      - train_classification.yaml / train_regression.yaml
      - execution_gpu.yaml
    """
    d = Path(directory)

    ds = DatasetSpec.model_validate(_load_yaml(d / dataset))
    sp = SplitSpec.model_validate(_load_yaml(d / split))
    em = EmbeddingSpec.model_validate(_load_yaml(d / embedding))
    tr = TrainSpec.model_validate(_load_yaml(d / train))
    ex = ExecutionSpec.model_validate(_load_yaml(d / execution))

    return {
        "dataset": ds,
        "split": sp,
        "embedding": em,
        "train": tr,
        "execution": ex,
    }


def validate_specs(
    *,
    dataset: DatasetSpec,
    split: SplitSpec,
    embedding: EmbeddingSpec,
    train: TrainSpec,
    execution: ExecutionSpec,
    strict: bool = True,
) -> Sequence[Any]:
    """
    Validate currently implemented checks (dataset, split, dataset↔split consistency).

    Returns
    -------
    issues : list[ValidationIssue]
    """
    issues = []
    issues += validate_dataset_spec(dataset)
    issues += validate_split_spec(split)
    issues += validate_dataset_split_consistency(dataset, split)

    issues = sort_issues(issues)

    if strict:
        errors = [i for i in issues if i.severity == "ERROR"]
        if errors:
            raise ValueError(format_issues(errors, header="MCS validation failed (ERROR issues):"))

    return issues


def create_record(
    *,
    dataset: DatasetSpec,
    split: SplitSpec,
    embedding: EmbeddingSpec,
    train: TrainSpec,
    execution: ExecutionSpec,
    artifacts: Optional[Sequence[ArtifactRef]] = None,
    metric_summary: Optional[MetricSummary] = None,
    notes: Optional[str] = None,
    salt: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ProvenanceRecord:
    return ProvenanceRecord.from_specs(
        dataset=dataset,
        split=split,
        embedding=embedding,
        train=train,
        execution=execution,
        artifacts=artifacts,
        metric_summary=metric_summary,
        notes=notes,
        salt=salt,
        extra=extra,
    )


def register_run(
    *,
    registry_root: str | Path = ".mcs_registry",
    dataset: DatasetSpec,
    split: SplitSpec,
    embedding: EmbeddingSpec,
    train: TrainSpec,
    execution: ExecutionSpec,
    record: ProvenanceRecord,
    artifact_mode: Literal["copy", "symlink", "reference"] = "reference",
    overwrite: bool = False,
) -> Dict[str, Any]:
    reg = LocalRegistry(registry_root)
    return reg.register_all(
        dataset=dataset,
        split=split,
        embedding=embedding,
        train=train,
        execution=execution,
        record=record,
        artifact_mode=artifact_mode,
        overwrite=overwrite,
    )


def run_pack(
    examples_dir: str | Path,
    *,
    train_file: str,
    execution_file: str,
    dataset_file: str = "dataset.yaml",
    split_file: str = "split.yaml",
    embedding_file: str = "embedding.yaml",
    registry_root: str | Path = ".mcs_registry",
    strict: bool = True,
    artifact_mode: Literal["copy", "symlink", "reference"] = "reference",
    overwrite: bool = False,
    artifacts: Optional[Sequence[ArtifactRef]] = None,
    metric_summary: Optional[MetricSummary] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-shot helper for notebooks: load → validate → record → register.

    Returns dict with:
      - specs
      - issues
      - record
      - registry_paths
    """
    specs = load_specs(
        examples_dir,
        dataset=dataset_file,
        split=split_file,
        embedding=embedding_file,
        train=train_file,
        execution=execution_file,
    )

    issues = validate_specs(**specs, strict=strict)

    record = create_record(
        **specs,
        artifacts=artifacts,
        metric_summary=metric_summary,
        notes=notes,
    )

    registry_paths = register_run(
        registry_root=registry_root,
        record=record,
        artifact_mode=artifact_mode,
        overwrite=overwrite,
        **specs,
    )

    return {
        "specs": specs,
        "issues": issues,
        "record": record,
        "registry_paths": registry_paths,
    }
