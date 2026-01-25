from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union
from pathlib import Path
import json
import shutil

from pydantic import BaseModel

from mcs.schemas.dataset import DatasetSpec
from mcs.schemas.split import SplitSpec
from mcs.schemas.embedding import EmbeddingSpec
from mcs.schemas.train import TrainSpec
from mcs.schemas.execution import ExecutionSpec

from mcs.provenance.record import ProvenanceRecord, ArtifactRef


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class LocalRegistry:
    """
    Minimal local registry for MCS v0.1.

    Layout
    ------
    .mcs_registry/
      specs/
        dataset/<fp>.json
        split/<fp>.json
        embedding/<fp>.json
        train/<fp>.json
        execution/<fp>.json
      runs/<run_id>.json
      artifacts/<run_id>/...
    """

    def __init__(self, root: Union[str, Path] = ".mcs_registry") -> None:
        self.root = Path(root)
        self.specs_dir = self.root / "specs"
        self.runs_dir = self.root / "runs"
        self.artifacts_dir = self.root / "artifacts"
        self._ensure_layout()

    # -------------------------
    # filesystem primitives
    # -------------------------

    def _ensure_layout(self) -> None:
        (self.specs_dir / "dataset").mkdir(parents=True, exist_ok=True)
        (self.specs_dir / "split").mkdir(parents=True, exist_ok=True)
        (self.specs_dir / "embedding").mkdir(parents=True, exist_ok=True)
        (self.specs_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.specs_dir / "execution").mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _dump_json(path: Path, data: Dict[str, Any], *, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    # -------------------------
    # spec registration
    # -------------------------

    def register_spec(
        self,
        kind: Literal["dataset", "split", "embedding", "train", "execution"],
        spec: BaseModel,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Store a canonical JSON form of the spec under specs/<kind>/<fingerprint>.json
        """
        fp = _get_fingerprint(spec)
        out = self.specs_dir / kind / f"{fp}.json"
        if out.exists() and not overwrite:
            return out
        data = _canonical_model_dump(spec)
        self._dump_json(out, data)
        return out

    def register_specs(
        self,
        *,
        dataset: DatasetSpec,
        split: SplitSpec,
        embedding: EmbeddingSpec,
        train: TrainSpec,
        execution: ExecutionSpec,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        return {
            "dataset": self.register_spec("dataset", dataset, overwrite=overwrite),
            "split": self.register_spec("split", split, overwrite=overwrite),
            "embedding": self.register_spec("embedding", embedding, overwrite=overwrite),
            "train": self.register_spec("train", train, overwrite=overwrite),
            "execution": self.register_spec("execution", execution, overwrite=overwrite),
        }

    # -------------------------
    # run registration
    # -------------------------

    def register_run(self, record: ProvenanceRecord, *, overwrite: bool = False) -> Path:
        """
        Store a ProvenanceRecord under runs/<run_id>.json
        """
        out = self.runs_dir / f"{record.run_id}.json"
        if out.exists() and not overwrite:
            return out
        self._dump_json(out, record.to_dict())
        return out

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load a run record dict from runs/<run_id>.json
        """
        p = self.runs_dir / f"{run_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found at {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    # -------------------------
    # artifact registration
    # -------------------------

    def register_artifact(
        self,
        run_id: str,
        artifact: ArtifactRef,
        *,
        mode: Literal["copy", "symlink", "reference"] = "reference",
        overwrite: bool = False,
    ) -> Path:
        """
        Register an artifact for a run.

        Parameters
        ----------
        mode:
          - "copy": copy file/dir into registry
          - "symlink": create symlink into registry (if filesystem supports)
          - "reference": do not copy; just return the intended target path (run artifacts dir)

        Notes
        -----
        - If artifact.path is not a local path (e.g., remote key), use mode="reference".
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        src = Path(artifact.path)
        # Target path uses kind as folder for readability
        target = run_dir / artifact.kind / src.name if src.name else (run_dir / artifact.kind / "artifact")
        target.parent.mkdir(parents=True, exist_ok=True)

        if mode == "reference":
            return target

        if not src.exists():
            raise FileNotFoundError(f"Artifact source not found: {src}")

        if target.exists():
            if overwrite:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            else:
                return target

        if mode == "copy":
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)
            return target

        if mode == "symlink":
            # On Windows / restricted FS this can fail; let it raise with clear message
            target.symlink_to(src.resolve(), target_is_directory=src.is_dir())
            return target

        raise ValueError(f"Unknown artifact registration mode: {mode}")

    def register_artifacts_from_record(
        self,
        record: ProvenanceRecord,
        *,
        mode: Literal["copy", "symlink", "reference"] = "reference",
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """
        Register all artifacts declared in a record.
        Returns mapping kind:path -> registry target.
        """
        out: Dict[str, Path] = {}
        for a in record.artifacts:
            tgt = self.register_artifact(record.run_id, a, mode=mode, overwrite=overwrite)
            out[f"{a.kind}:{a.path}"] = tgt
        return out

    # -------------------------
    # one-shot helper
    # -------------------------

    def register_all(
        self,
        *,
        dataset: DatasetSpec,
        split: SplitSpec,
        embedding: EmbeddingSpec,
        train: TrainSpec,
        execution: ExecutionSpec,
        record: ProvenanceRecord,
        artifact_mode: Literal["copy", "symlink", "reference"] = "reference",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        One-shot helper: register specs + run + artifacts.
        """
        spec_paths = self.register_specs(
            dataset=dataset, split=split, embedding=embedding, train=train, execution=execution, overwrite=overwrite
        )
        run_path = self.register_run(record, overwrite=overwrite)
        artifact_paths = self.register_artifacts_from_record(record, mode=artifact_mode, overwrite=overwrite)
        return {
            "specs": spec_paths,
            "run": run_path,
            "artifacts": artifact_paths,
        }


# -------------------------
# internal helpers
# -------------------------

def _canonical_model_dump(spec: BaseModel) -> Dict[str, Any]:
    # We rely on pydantic's JSON mode for dates etc; exclude_none for canonicality
    return spec.model_dump(mode="json", exclude_none=True)


def _get_fingerprint(spec: BaseModel) -> str:
    # All your specs implement fingerprint(); keep fallback for safety
    if hasattr(spec, "fingerprint") and callable(getattr(spec, "fingerprint")):
        return str(spec.fingerprint())
    # Fallback: hash stable dump (should never be used in your project)
    return _sha256(_stable_json(_canonical_model_dump(spec)))


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
