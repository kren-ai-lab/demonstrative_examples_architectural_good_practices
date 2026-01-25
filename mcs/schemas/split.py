from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import hashlib

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

from mcs.version import get_mcs_version

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LeakageCheck(BaseModel):
    """
    Declarative leakage check.

    Examples
    --------
    - {"type": "sequence_identity", "threshold": 0.3, "scope": "train_test"}
    - {"type": "group_overlap", "field": "cluster_id", "scope": "all"}
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "sequence_identity",
        "embedding_similarity",
        "group_overlap",
        "exact_duplicate",
        "custom",
    ] = Field(..., description="Leakage check type.")
    scope: Literal["train_test", "train_val", "val_test", "all"] = Field(
        default="all", description="Which partitions to compare."
    )
    threshold: Optional[float] = Field(
        default=None, description="Threshold in [0,1] when applicable."
    )
    field: Optional[str] = Field(
        default=None, description="Grouping field for group_overlap checks."
    )
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_check(self) -> "LeakageCheck":
        if self.type in {"sequence_identity", "embedding_similarity"}:
            if self.threshold is None:
                raise ValueError(f"LeakageCheck.threshold is required when type='{self.type}'.")
            if not (0.0 <= float(self.threshold) <= 1.0):
                raise ValueError("LeakageCheck.threshold must be within [0, 1].")
        if self.type == "group_overlap" and not self.field:
            raise ValueError("LeakageCheck.field is required when type='group_overlap'.")
        return self


class SplitRatios(BaseModel):
    """Train/val/test ratios."""
    model_config = ConfigDict(extra="forbid")

    train: float = Field(..., gt=0.0, lt=1.0)
    val: float = Field(..., ge=0.0, lt=1.0)
    test: float = Field(..., gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "SplitRatios":
        s = float(self.train) + float(self.val) + float(self.test)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0 (got {s}).")
        return self


class IdentityDefinition(BaseModel):
    """
    Defines how similarity/identity is measured for splitting regimes.

    Notes
    -----
    'measure' and 'threshold' are declared for interpretability and reproducibility.
    The MCS does not prescribe a specific implementation.
    """
    model_config = ConfigDict(extra="forbid")

    measure: Literal["sequence_identity", "embedding_similarity", "cluster_label", "time", "structure"] = Field(
        ..., description="Identity/similarity basis."
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold in [0,1] for similarity-based regimes (when applicable).",
    )
    tool: Optional[str] = Field(
        default=None,
        description="Optional tool/method used to compute identity (mmseqs2, blast, custom).",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_threshold(self) -> "IdentityDefinition":
        if self.measure in {"sequence_identity", "embedding_similarity"}:
            if self.threshold is None:
                raise ValueError("identity.threshold is required for similarity-based measures.")
            if not (0.0 <= float(self.threshold) <= 1.0):
                raise ValueError("identity.threshold must be within [0,1].")
        return self


class SplitSpec(BaseModel):
    """
    Minimal Community Standard (MCS) v0.1 Split Specification.

    Purpose
    -------
    Make partition semantics explicit: protocol, intended generalisation regime,
    identity definition, ratios, seeds, and leakage checks.
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version)

    split_name: str = Field(..., min_length=1, description="Name/version tag for this split.")
    protocol: Literal[
        "random",
        "stratified",
        "group",
        "cluster_aware",
        "temporal",
        "structure_aware",
        "custom",
    ] = Field(..., description="Splitting protocol / generalisation regime.")

    ratios: SplitRatios = Field(..., description="Train/val/test ratios.")
    seed: int = Field(..., description="Seed for deterministic split generation.")

    # Optional fields depending on protocol
    stratify_field: Optional[str] = Field(default=None, description="Label/field used for stratification.")
    group_field: Optional[str] = Field(default=None, description="Grouping field for group-based splitting.")
    identity: Optional[IdentityDefinition] = Field(default=None, description="Identity definition for cluster-aware/structure-aware/temporal.")
    time_field: Optional[str] = Field(default=None, description="Field name containing timestamps for temporal splits.")

    leakage_checks: List[LeakageCheck] = Field(default_factory=list)

    notes: Optional[str] = Field(default=None)

    @field_validator("mcs_version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mcs_version must be a non-empty string.")
        return v

    @model_validator(mode="after")
    def _validate_protocol_requirements(self) -> "SplitSpec":
        # protocol-specific requirements
        if self.protocol == "stratified" and not self.stratify_field:
            raise ValueError("stratify_field is required when protocol='stratified'.")

        if self.protocol in {"group"} and not self.group_field:
            raise ValueError("group_field is required when protocol='group'.")

        if self.protocol in {"cluster_aware", "structure_aware"}:
            if self.identity is None:
                raise ValueError(f"identity is required when protocol='{self.protocol}'.")
            # cluster_aware typically relies on similarity or cluster labels
            if self.protocol == "cluster_aware" and self.identity.measure not in {
                "sequence_identity", "embedding_similarity", "cluster_label"
            }:
                raise ValueError("cluster_aware expects identity.measure in {sequence_identity, embedding_similarity, cluster_label}.")

        if self.protocol == "temporal":
            if not self.time_field:
                raise ValueError("time_field is required when protocol='temporal'.")
            # identity optional here; time_field is the key declaration

        # At least one leakage check is strongly encouraged; enforce for non-random regimes
        if self.protocol != "random" and len(self.leakage_checks) == 0:
            raise ValueError("leakage_checks must be provided for non-random split protocols (to ensure auditability).")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        """Stable fingerprint (SHA256) of canonical JSON representation."""
        return _sha256(_stable_json(self.to_dict()))

    def describe_regime(self) -> str:
        """
        Short human-readable description for paper tables/boxes.
        """
        base = f"{self.protocol} split"
        if self.protocol == "stratified":
            return f"{base} stratified by '{self.stratify_field}'"
        if self.protocol == "group":
            return f"{base} grouped by '{self.group_field}'"
        if self.protocol in {"cluster_aware", "structure_aware"} and self.identity:
            t = self.identity.threshold
            if t is None:
                return f"{base} using {self.identity.measure}"
            return f"{base} using {self.identity.measure} (threshold={t})"
        if self.protocol == "temporal":
            return f"{base} ordered by '{self.time_field}'"
        return base
