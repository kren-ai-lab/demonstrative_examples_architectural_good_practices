from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import json
import hashlib
from mcs.version import get_mcs_version

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CrossValidationSpec(BaseModel):
    """Optional cross-validation declaration."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    folds: int = Field(default=5, ge=2, description="Number of folds if enabled.")
    strategy: Literal["kfold", "stratified_kfold", "group_kfold", "time_series", "custom"] = Field(default="kfold")
    group_field: Optional[str] = Field(default=None, description="Group field name when using group-aware CV.")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CalibrationSpec(BaseModel):
    """Optional calibration declaration (kept lightweight)."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    method: Literal["platt", "isotonic", "temperature", "custom"] = Field(default="platt")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TrainOutput(BaseModel):
    """Logical outputs for training runs (model artifacts, metrics, predictions)."""
    model_config = ConfigDict(extra="forbid")

    model_path: Optional[str] = Field(default=None, description="Path/artifact key for trained model.")
    metrics_path: Optional[str] = Field(default=None, description="Path/artifact key for metrics.")
    predictions_path: Optional[str] = Field(default=None, description="Path/artifact key for predictions.")
    checkpoint_dir: Optional[str] = Field(default=None, description="Optional checkpoint directory.")


class TrainSpec(BaseModel):
    """
    Minimal Community Standard (MCS) v0.1 Training Specification.

    Purpose
    -------
    Declare the training/evaluation intent and hyperparameters without prescribing
    a specific ML framework.
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version, description="MCS schema version.")

    name: str = Field(..., min_length=1, description="Training configuration name/tag.")
    task: Literal["classification", "regression", "ranking", "unsupervised"] = Field(...)

    model_family: Literal["linear", "rf", "svm", "xgb", "nn", "gdl", "custom"] = Field(
        ..., description="High-level model family label."
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional specific model identifier (e.g., 'RandomForestClassifier', 'MLP-2x512').",
    )

    hyperparams: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters (must be JSON-serializable).")
    metrics: List[str] = Field(default_factory=list, description="Metric names (e.g., accuracy, rmse).")

    seed: int = Field(default=42, description="Seed for deterministic training where applicable.")
    early_stopping: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional early stopping configuration."
    )

    cross_validation: Optional[CrossValidationSpec] = Field(default=None)
    calibration: Optional[CalibrationSpec] = Field(default=None)

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form parameters (e.g., class weights, loss options, post-processing).",
    )

    input_split_fingerprint: Optional[str] = Field(default=None, description="Optional SplitSpec fingerprint.")
    input_embedding_fingerprint: Optional[str] = Field(default=None, description="Optional EmbeddingSpec fingerprint.")

    output: Optional[TrainOutput] = Field(default=None)

    @field_validator("mcs_version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mcs_version must be a non-empty string.")
        return v

    @model_validator(mode="after")
    def _validate_minimal(self) -> "TrainSpec":
        # Metrics are encouraged except for unsupervised
        if self.task != "unsupervised" and len(self.metrics) == 0:
            raise ValueError("metrics must be provided for supervised tasks (classification/regression/ranking).")

        # Calibration should only be used for classification (by default)
        if self.calibration and self.calibration.enabled and self.task != "classification":
            raise ValueError("calibration.enabled is only valid when task='classification'.")

        # Cross-validation group_field requirement
        if self.cross_validation and self.cross_validation.enabled:
            if self.cross_validation.strategy == "group_kfold" and not self.cross_validation.group_field:
                raise ValueError("cross_validation.group_field is required when strategy='group_kfold'.")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        return _sha256(_stable_json(self.to_dict()))
