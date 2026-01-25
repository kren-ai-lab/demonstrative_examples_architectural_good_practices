from __future__ import annotations

from typing import Any, Dict, Literal, Optional
import json
import hashlib
from mcs.version import get_mcs_version

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TokenizationSpec(BaseModel):
    """Tokenization and sequence handling settings."""
    model_config = ConfigDict(extra="forbid")

    max_length: int = Field(default=1024, ge=1, description="Maximum sequence length after processing.")
    truncation: bool = Field(default=True, description="Whether to truncate sequences longer than max_length.")
    padding: Literal["none", "max_length", "longest"] = Field(
        default="none", description="Padding strategy."
    )
    add_special_tokens: bool = Field(default=True, description="Whether to add special tokens (model-dependent).")


class EmbeddingOutput(BaseModel):
    """Logical output declaration for embeddings."""
    model_config = ConfigDict(extra="forbid")

    format: Literal["npy", "npz", "pt", "parquet", "hdf5", "other"] = Field(
        default="npz", description="Materialized embeddings format."
    )
    path: str = Field(..., min_length=1, description="Relative/absolute path or artifact key.")
    checksum: Optional[str] = Field(default=None, description="Optional checksum of the embedding file.")


class EmbeddingSpec(BaseModel):
    """
    Minimal Community Standard (MCS) v0.1 Embedding Specification.

    Purpose
    -------
    Externalize representation extraction choices to prevent silent representational drift.
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version, description="MCS schema version.")

    name: str = Field(..., min_length=1, description="Embedding configuration name/tag.")
    model_id: str = Field(..., min_length=1, description="Model identifier (e.g., esm2_t33_650M_UR50D).")
    provider: Optional[Literal["huggingface", "facebook", "local", "custom"]] = Field(
        default=None, description="Optional provider label."
    )

    tokenization: TokenizationSpec = Field(default_factory=TokenizationSpec)

    # Layer + aggregation
    layer: Literal["last"] | int = Field(
        default="last",
        description="Layer index or 'last'. If int, must be >= 0.",
    )
    pooling: Literal["mean", "cls", "sum", "attention", "custom"] = Field(
        default="mean", description="Pooling strategy to obtain fixed-size embeddings."
    )

    # Numerical / compute properties (declared for reproducibility, not enforced)
    dtype: Literal["fp32", "fp16", "bf16"] = Field(default="fp32")
    device: Literal["cpu", "cuda", "mps", "tpu", "other"] = Field(default="cpu")
    batch_size: int = Field(default=8, ge=1)
    seed: Optional[int] = Field(default=None, description="Optional seed if extraction includes stochasticity.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form parameters (e.g., attention pooling options, custom post-processing).",
    )

    # Optional linkage to inputs
    input_dataset_fingerprint: Optional[str] = Field(
        default=None,
        description="Optional DatasetSpec fingerprint this embedding was derived from.",
    )

    output: Optional[EmbeddingOutput] = Field(default=None)

    @field_validator("mcs_version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mcs_version must be a non-empty string.")
        return v

    @field_validator("layer")
    @classmethod
    def _layer_non_negative_if_int(cls, v: int | str) -> int | str:
        if isinstance(v, int) and v < 0:
            raise ValueError("layer must be >= 0 when provided as an integer.")
        return v

    @model_validator(mode="after")
    def _validate_minimal(self) -> "EmbeddingSpec":
        # If padding is max_length, truncation should typically be True (warn-level here -> we encode as error only if inconsistent)
        if self.tokenization.padding == "max_length" and not self.tokenization.truncation:
            raise ValueError("tokenization.truncation should be True when padding='max_length' to ensure bounded length.")

        # Output hygiene
        if self.output is not None and not self.output.path.strip():
            raise ValueError("output.path cannot be empty when output is provided.")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        return _sha256(_stable_json(self.to_dict()))
