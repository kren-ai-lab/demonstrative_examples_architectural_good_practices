from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from datetime import date
import json
import hashlib
from mcs.version import get_mcs_version

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

def _stable_json(obj: Any) -> str:
    """Return a canonical JSON string (stable ordering) for hashing/fingerprints."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DatasetSource(BaseModel):
    """
    A dataset provenance source.

    Notes
    -----
    Keep this minimal and tool-agnostic. 'type' is a controlled label for readability,
    not an exhaustive ontology.
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal["database", "repository", "publication", "internal", "url"] = Field(
        ..., description="Origin type of the data."
    )
    name: str = Field(..., min_length=1, description="Source name (e.g., UniProt, PDB, custom lab DB).")
    identifier: Optional[str] = Field(
        default=None,
        description="Stable identifier (e.g., DOI, accession set ID, Zenodo record, commit hash).",
    )
    version: Optional[str] = Field(
        default=None,
        description="Explicit version tag (release, snapshot date, or commit).",
    )
    accessed: Optional[date] = Field(
        default=None,
        description="Access date for mutable sources.",
    )
    uri: Optional[str] = Field(
        default=None,
        description="Resolvable URI when applicable.",
    )


class FilterRule(BaseModel):
    """
    Declarative filtering rule.

    Examples
    --------
    - {"field": "sequence_length", "op": "between", "value": [30, 500]}
    - {"field": "organism", "op": "in", "value": ["Homo sapiens", "Escherichia coli"]}
    - {"field": "has_noncanonical", "op": "eq", "value": False}
    """
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., min_length=1, description="Target field name.")
    op: Literal[
        "eq", "ne", "lt", "le", "gt", "ge",
        "in", "not_in",
        "contains", "not_contains",
        "between",
        "regex",
        "exists",
    ] = Field(..., description="Operation.")
    value: Any = Field(default=None, description="Operation value. Type depends on 'op'.")
    description: Optional[str] = Field(default=None, description="Human-readable rationale.")


class RedundancyControl(BaseModel):
    """
    Redundancy control / deduplication policy.

    Notes
    -----
    The goal is to *declare* what was done, not enforce a specific tool.
    """
    model_config = ConfigDict(extra="forbid")

    method: Literal["none", "sequence_identity", "cluster", "custom"] = Field(
        ..., description="Redundancy control method."
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold in [0,1] when method requires it (e.g., seq identity).",
    )
    tool: Optional[str] = Field(
        default=None,
        description="Tool name when relevant (e.g., mmseqs2, cd-hit).",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form parameters for the chosen tool/method.",
    )

    @model_validator(mode="after")
    def _check_threshold(self) -> "RedundancyControl":
        if self.method in {"sequence_identity", "cluster"}:
            if self.threshold is None:
                raise ValueError(f"redundancy_control.threshold is required when method='{self.method}'.")
            if not (0.0 <= float(self.threshold) <= 1.0):
                raise ValueError("redundancy_control.threshold must be within [0, 1].")
        return self


class LabelField(BaseModel):
    """Label field schema (single- or multi-label)."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    dtype: Literal["float", "int", "bool", "str", "category"] = Field(..., description="Label type.")
    description: Optional[str] = Field(default=None)
    unit: Optional[str] = Field(default=None)


class MetadataField(BaseModel):
    """Metadata field schema for downstream interoperability."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    dtype: Literal["float", "int", "bool", "str", "category"] = Field(...)
    required: bool = Field(default=False)
    description: Optional[str] = Field(default=None)


class DatasetOutput(BaseModel):
    """Logical output declaration (no I/O enforced here)."""
    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "parquet", "jsonl", "fasta", "hdf5", "other"] = Field(
        ..., description="Materialized dataset format."
    )
    path: str = Field(..., min_length=1, description="Relative or absolute path, or artifact key.")
    checksum: Optional[str] = Field(
        default=None,
        description="Optional checksum of the materialized dataset file.",
    )


class DatasetSpec(BaseModel):
    """
    Minimal Community Standard (MCS) v0.1 Dataset Specification.

    Purpose
    -------
    Externalize the minimal assumptions required to reconstruct and interpret
    a curated dataset: provenance, filtering, redundancy control, and schema.
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version, description="MCS schema version.")

    name: str = Field(..., min_length=1, description="Dataset name.")
    description: Optional[str] = Field(default=None, description="Short description.")
    domain: Optional[str] = Field(default="protein", description="Domain label (e.g., protein, peptide).")
    created_by: Optional[str] = Field(default=None, description="Creator or lab.")
    created_at: Optional[date] = Field(default=None, description="Creation date.")

    sources: List[DatasetSource] = Field(default_factory=list, description="Provenance sources.")
    filters: List[FilterRule] = Field(default_factory=list, description="Declarative filters.")
    redundancy_control: RedundancyControl = Field(
        default_factory=lambda: RedundancyControl(method="none"),
        description="Redundancy policy.",
    )

    labels: List[LabelField] = Field(default_factory=list, description="Label schema (can be empty for unlabeled).")
    metadata: List[MetadataField] = Field(default_factory=list, description="Metadata fields.")

    sequence_field: str = Field(default="sequence", description="Field name holding the sequence.")
    id_field: str = Field(default="id", description="Primary identifier field name.")

    output: Optional[DatasetOutput] = Field(default=None, description="Output materialization info.")

    @field_validator("mcs_version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mcs_version must be a non-empty string.")
        return v

    @model_validator(mode="after")
    def _validate_minimal(self) -> "DatasetSpec":
        # Must have at least one source for provenance (core MCS intent)
        if len(self.sources) == 0:
            raise ValueError("sources must contain at least one provenance entry.")

        # Enforce uniqueness of label/metadata names
        label_names = [l.name for l in self.labels]
        if len(label_names) != len(set(label_names)):
            raise ValueError("labels contain duplicated 'name' entries.")

        meta_names = [m.name for m in self.metadata]
        if len(meta_names) != len(set(meta_names)):
            raise ValueError("metadata contain duplicated 'name' entries.")

        # Disallow overlaps between id/sequence and metadata/labels to avoid ambiguity
        reserved = {self.id_field, self.sequence_field}
        if any(n in reserved for n in label_names):
            raise ValueError("labels cannot use reserved field names (id_field/sequence_field).")
        if any(n in reserved for n in meta_names):
            raise ValueError("metadata cannot use reserved field names (id_field/sequence_field).")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict with stable keys (via model_dump)."""
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        """
        Stable fingerprint for the spec content.

        Notes
        -----
        This hashes the canonical JSON representation of the spec.
        """
        return _sha256(_stable_json(self.to_dict()))
