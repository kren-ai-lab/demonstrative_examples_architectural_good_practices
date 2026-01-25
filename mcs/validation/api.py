from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import yaml

from mcs.schemas.dataset import DatasetSpec
from mcs.schemas.split import SplitSpec
from mcs.schemas.embedding import EmbeddingSpec
from mcs.schemas.train import TrainSpec
from mcs.schemas.execution import ExecutionSpec

from mcs.validation.types import ValidationIssue
from mcs.validation.dataset_validator import validate_dataset_spec
from mcs.validation.split_validator import validate_split_spec
from mcs.validation.embedding_validator import validate_embedding_spec
from mcs.validation.train_validator import validate_train_spec
from mcs.validation.execution_validator import validate_execution_spec

# Unified consistency layer (single file)
from mcs.validation.consistency import (
    validate_dataset_split_consistency,  # kept for backward compatibility
    validate_consistency,                # unified multi-spec checks
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at '{path}' must contain a mapping (dict-like) at the top level.")
    return data


def sort_issues(issues: Sequence[ValidationIssue]) -> List[ValidationIssue]:
    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    return sorted(issues, key=lambda x: (order.get(x.severity, 9), x.code, x.path or ""))


def format_issues(issues: Sequence[ValidationIssue], header: Optional[str] = None) -> str:
    """
    Human-readable formatter, suitable for logging and notebooks.
    """
    lines: List[str] = []
    if header:
        lines.append(header)

    for i in issues:
        line = f"- [{i.severity}] {i.code}: {i.message}"
        if i.path:
            line += f" (path={i.path})"
        if i.hint:
            line += f" | hint: {i.hint}"
        lines.append(line)

    return "\n".join(lines)


def summarize_by_severity(issues: Sequence[ValidationIssue]) -> Dict[str, int]:
    """
    Quick summary counts for dashboards / notebook display.
    """
    out = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for i in issues:
        if i.severity in out:
            out[i.severity] += 1
    return out


# ---------------------------------------------------------------------
# 2-spec API (dataset + split) — backwards compatible
# ---------------------------------------------------------------------

def validate_all(
    dataset: DatasetSpec,
    split: SplitSpec,
    *,
    strict: bool = True,
) -> List[ValidationIssue]:
    """
    Run dataset + split + cross-spec consistency validations.

    Parameters
    ----------
    strict
        If True, raises ValueError if any ERROR issues exist.

    Returns
    -------
    List[ValidationIssue]
        All issues (ERROR/WARNING/INFO), ordered by severity then code.
    """
    issues: List[ValidationIssue] = []
    issues.extend(validate_dataset_spec(dataset))
    issues.extend(validate_split_spec(split))

    # Use unified consistency function (works with only dataset+split)
    issues.extend(validate_consistency(dataset, split))

    issues = sort_issues(issues)

    if strict:
        errors = [i for i in issues if i.severity == "ERROR"]
        if errors:
            raise ValueError(format_issues(errors, header="MCS validation failed (ERROR issues):"))

    return issues


def validate_from_files(
    dataset_yaml: str,
    split_yaml: str,
    *,
    strict: bool = True,
) -> List[ValidationIssue]:
    """Convenience: load YAML specs and validate (dataset + split)."""
    ds = DatasetSpec.model_validate(load_yaml(dataset_yaml))
    sp = SplitSpec.model_validate(load_yaml(split_yaml))
    return validate_all(ds, sp, strict=strict)


# ---------------------------------------------------------------------
# 5-spec API (dataset + split + embedding + train + execution)
# ---------------------------------------------------------------------

def validate_all_5specs(
    dataset: DatasetSpec,
    split: SplitSpec,
    embedding: EmbeddingSpec,
    train: TrainSpec,
    execution: ExecutionSpec,
    *,
    strict: bool = True,
) -> List[ValidationIssue]:
    """
    Run per-spec semantic validations and unified cross-spec consistency
    across dataset/split/embedding/train/execution.

    Parameters
    ----------
    strict
        If True, raises ValueError if any ERROR issues exist.
    """
    issues: List[ValidationIssue] = []

    # Per-spec semantic validations
    issues.extend(validate_dataset_spec(dataset))
    issues.extend(validate_split_spec(split))
    issues.extend(validate_embedding_spec(embedding))
    issues.extend(validate_train_spec(train))
    issues.extend(validate_execution_spec(execution))

    # Unified consistency across all provided specs
    issues.extend(validate_consistency(dataset, split, embedding=embedding, train=train, execution=execution))

    issues = sort_issues(issues)

    if strict:
        errors = [i for i in issues if i.severity == "ERROR"]
        if errors:
            raise ValueError(format_issues(errors, header="MCS validation failed (ERROR issues):"))

    return issues


def validate_5specs_from_files(
    dataset_yaml: str,
    split_yaml: str,
    embedding_yaml: str,
    train_yaml: str,
    execution_yaml: str,
    *,
    strict: bool = True,
) -> List[ValidationIssue]:
    """Convenience: load YAML specs and validate (5-spec pack)."""
    ds = DatasetSpec.model_validate(load_yaml(dataset_yaml))
    sp = SplitSpec.model_validate(load_yaml(split_yaml))
    em = EmbeddingSpec.model_validate(load_yaml(embedding_yaml))
    tr = TrainSpec.model_validate(load_yaml(train_yaml))
    ex = ExecutionSpec.model_validate(load_yaml(execution_yaml))
    return validate_all_5specs(ds, sp, em, tr, ex, strict=strict)
