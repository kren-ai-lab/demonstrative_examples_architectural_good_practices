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


class DependencySpec(BaseModel):
    """
    Dependency declaration.

    Notes
    -----
    We do not prescribe a specific lockfile manager; instead we allow
    declaring one or more reproducibility anchors.
    """
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["lockfile", "pip_freeze", "conda_env", "poetry_lock", "uv_lock", "custom"] = Field(
        default="pip_freeze"
    )
    reference: str = Field(
        ...,
        min_length=1,
        description="Path, URI, or hash reference to the dependency snapshot (e.g., requirements.txt hash).",
    )
    checksum: Optional[str] = Field(default=None, description="Optional checksum of the dependency snapshot.")


class HardwareSpec(BaseModel):
    """Minimal hardware declaration."""
    model_config = ConfigDict(extra="forbid")

    cpu: Optional[str] = Field(default=None, description="CPU model/name.")
    gpu: Optional[str] = Field(default=None, description="GPU model/name.")
    accelerator: Optional[Literal["cuda", "mps", "tpu", "none", "other"]] = Field(default="none")
    ram_gb: Optional[float] = Field(default=None, ge=0.0)
    notes: Optional[str] = Field(default=None)


class ContainerSpec(BaseModel):
    """Optional container/environment encapsulation."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["docker", "singularity", "apptainer", "none", "other"] = Field(default="none")
    image: Optional[str] = Field(default=None, description="Container image name/tag.")
    digest: Optional[str] = Field(default=None, description="Immutable digest (recommended).")


class ExecutionSpec(BaseModel):
    """
    Minimal Community Standard (MCS) v0.1 Execution Specification.

    Purpose
    -------
    Declare the execution environment to make runs auditable and comparable.
    """
    model_config = ConfigDict(extra="forbid")

    mcs_version: str = Field(default_factory=get_mcs_version, description="MCS schema version.")

    name: str = Field(..., min_length=1, description="Execution profile name/tag.")
    python_version: str = Field(..., min_length=1, description="Python version string, e.g., '3.12.1'.")
    platform: Optional[str] = Field(default=None, description="OS/platform string, e.g., Linux-x86_64.")
    git_commit: Optional[str] = Field(default=None, description="Git commit hash of the code used.")
    working_directory: Optional[str] = Field(default=None, description="Working directory (optional).")

    dependencies: DependencySpec = Field(..., description="Dependency snapshot reference.")
    hardware: Optional[HardwareSpec] = Field(default=None)
    container: Optional[ContainerSpec] = Field(default=None)

    determinism_flags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form determinism flags (e.g., cudnn.deterministic, torch.use_deterministic_algorithms).",
    )

    notes: Optional[str] = Field(default=None)

    @field_validator("mcs_version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mcs_version must be a non-empty string.")
        return v

    @model_validator(mode="after")
    def _validate_minimal(self) -> "ExecutionSpec":
        # If container is declared with docker/singularity, encourage immutable digest by requiring digest when image is provided
        if self.container and self.container.type not in {"none"}:
            if self.container.image and not self.container.digest:
                raise ValueError("container.digest is recommended and required when container.image is provided (immutability).")

        # git_commit optional but strongly recommended; keep as INFO in validator layer later, not error here
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        return _sha256(_stable_json(self.to_dict()))
