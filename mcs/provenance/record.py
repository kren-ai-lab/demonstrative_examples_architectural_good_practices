from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, model_validator

from mcs.version import get_mcs_version
from mcs.schemas.dataset import DatasetSpec
from mcs.schemas.split import SplitSpec
from mcs.schemas.embedding import EmbeddingSpec
from mcs.schemas.train import TrainSpec
from mcs.schemas.execution import ExecutionSpec


# ----------------------------
# helpers (stable hashing)
# ----------------------------

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ----------------------------
# provenance building blocks
# ----------------------------

class ArtifactRef(BaseModel):
    """
    A lightweight reference to a produced artifact.

    Notes
    -----
    - 'path' may be a file path, directory, or logical artifact key.
    - 'checksum' is optional but recommended for file-based reproducibility.
    """
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., min_length=1, description="Artifact kind (e.g., embeddings, model, metrics, predictions, figure).")
    path: str = Field(..., min_length=1, description="Path or artifact key.")
    format: Optional[str] = Field(default=None, description="Format (e.g., npz, parquet, json).")
    checksum: Optional[str] = Field(default=None, description="Optional checksum of the artifact.")
    description: Optional[str] = Field(default=None, description="Optional free-text description.")


class MetricSummary(BaseModel):
    """
    Minimal metric summary for quick inspection.

    Notes
    -----
    Keep this small. Full metrics should live in an artifact (metrics.json).
    """
    model_config = ConfigDict(extra="forbid")

    split: Optional[Literal["train", "val", "test", "cv", "unknown"]] = Field(default="unknown")
    metrics: Dict[str, float] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """
    MCS v0.1 ProvenanceRecord.

    Purpose
    -------
    Provide an auditable, deterministic linkage between:
      (i) Specs (Dataset/Split/Embedding/Train/Execution),
      (ii) Produced artifacts,
      (iii) Minimal run context.

    Core design goals:
    - deterministic run_id derived from spec fingerprints (+ optional salt)
    - serializable record (JSON/YAML-friendly)
    - minimal but extensible
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version)

    # Deterministic identifiers
    run_id: str = Field(..., min_length=16, description="Deterministic run identifier (sha256).")
    created_at_utc: str = Field(default_factory=_utc_now_iso)

    # Spec fingerprints (load-bearing)
    dataset_fingerprint: str
    split_fingerprint: str
    embedding_fingerprint: str
    train_fingerprint: str
    execution_fingerprint: str

    # Optional helpful tags
    dataset_name: Optional[str] = None
    split_name: Optional[str] = None
    embedding_name: Optional[str] = None
    train_name: Optional[str] = None
    execution_name: Optional[str] = None

    # Outputs
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    metric_summary: Optional[MetricSummary] = Field(default=None)

    # Free-form extensibility (kept explicit)
    notes: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fps(self) -> "ProvenanceRecord":
        fps = [
            self.dataset_fingerprint,
            self.split_fingerprint,
            self.embedding_fingerprint,
            self.train_fingerprint,
            self.execution_fingerprint,
        ]
        if any((not f) or (not str(f).strip()) for f in fps):
            raise ValueError("All spec fingerprints must be non-empty strings.")
        return self

    # ----------------------------
    # constructors
    # ----------------------------

    @classmethod
    def compute_run_id(
        cls,
        *,
        dataset_fp: str,
        split_fp: str,
        embedding_fp: str,
        train_fp: str,
        execution_fp: str,
        salt: Optional[str] = None,
    ) -> str:
        """
        Deterministic run_id from the five spec fingerprints (+ optional salt).

        Notes
        -----
        - If you want multiple runs with the same specs (e.g., different random seeds outside specs),
          you can pass a salt. Prefer making seeds part of specs though.
        """
        payload = {
            "dataset": dataset_fp,
            "split": split_fp,
            "embedding": embedding_fp,
            "train": train_fp,
            "execution": execution_fp,
            "salt": salt,
        }
        return _sha256(_stable_json(payload))

    @classmethod
    def from_specs(
        cls,
        *,
        dataset: DatasetSpec,
        split: SplitSpec,
        embedding: EmbeddingSpec,
        train: TrainSpec,
        execution: ExecutionSpec,
        artifacts: Optional[Sequence[ArtifactRef]] = None,
        metric_summary: Optional[MetricSummary] = None,
        salt: Optional[str] = None,
        notes: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "ProvenanceRecord":
        ds_fp = dataset.fingerprint()
        sp_fp = split.fingerprint()
        em_fp = embedding.fingerprint()
        tr_fp = train.fingerprint()
        ex_fp = execution.fingerprint()

        run_id = cls.compute_run_id(
            dataset_fp=ds_fp,
            split_fp=sp_fp,
            embedding_fp=em_fp,
            train_fp=tr_fp,
            execution_fp=ex_fp,
            salt=salt,
        )

        return cls(
            run_id=run_id,
            dataset_fingerprint=ds_fp,
            split_fingerprint=sp_fp,
            embedding_fingerprint=em_fp,
            train_fingerprint=tr_fp,
            execution_fingerprint=ex_fp,
            dataset_name=getattr(dataset, "name", None),
            split_name=getattr(split, "split_name", None),
            embedding_name=getattr(embedding, "name", None),
            train_name=getattr(train, "name", None),
            execution_name=getattr(execution, "name", None),
            artifacts=list(artifacts) if artifacts else [],
            metric_summary=metric_summary,
            notes=notes,
            extra=extra or {},
        )

    # ----------------------------
    # serialization helpers
    # ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        """
        Fingerprint of the record itself (separate from run_id).
        """
        return _sha256(_stable_json(self.to_dict()))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)

    def save_json(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path

    # ----------------------------
    # convenience
    # ----------------------------

    def add_artifact(
        self,
        *,
        kind: str,
        path: str,
        format: Optional[str] = None,
        checksum: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self.artifacts.append(
            ArtifactRef(kind=kind, path=path, format=format, checksum=checksum, description=description)
        )
