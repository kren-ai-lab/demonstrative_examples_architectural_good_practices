#!/usr/bin/env python3
"""
MCS Case Study — Script 3: Embed split pools with Sylphy (v2 folder layout)

Folder layout expected (as in your repo screenshot):

case_demo_01/
  dataset/
    amp_raw_10k.csv               # (not required by this script, but kept for completeness)
  schemas/
    dataset.yaml                  # dataset manifest
  splits/
    random/seed_<SEED>/{train.csv,val.csv,test.csv,split.yaml,...}
    stratified/seed_<SEED>/{train.csv,val.csv,test.csv,split.yaml,...}

This script iterates over all split folders and extracts embeddings using Sylphy.

Defaults requested:
- pooling: mean (average)
- layer: last layer (uses Sylphy default; attempts to set -1 if supported)
- batch_size: 4
- memory cleanup between splits (gc + torch.cuda.empty_cache)

Outputs (per seed folder):
- embedding_<MODEL_SHORT>_mean_last_bs4/
    - train.parquet / val.parquet / test.parquet
    - schema.yaml
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

try:
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover
    pkg_version = None

# Optional torch cleanup
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from sylphy.embedding_extractor import create_embedding  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_")


def cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_set_last_layer(embedder) -> Dict[str, object]:
    """
    Sylphy layer selection argument names may vary by version.
    We default to library behavior (typically last layer) and try to set -1 if possible.
    """
    meta: Dict[str, object] = {"layer_selection": "default(last)"}
    for attr in ("layer", "layer_idx", "layer_index", "layers"):
        if hasattr(embedder, attr):
            try:
                setattr(embedder, attr, -1)
                meta["layer_selection"] = f"attr:{attr}=-1"
                return meta
            except Exception:
                pass
    return meta


# -----------------------------
# Schema
# -----------------------------
@dataclass
class EmbeddingSchema:
    mcs_version: str
    created_utc: str
    dataset_id: str
    split_id: str
    model_name: str
    model_short: str
    device: str
    precision: str
    batch_size: int
    pooling: str
    max_length: Optional[int]
    layer_selection: str
    sylphy_version: Optional[str]
    inputs: Dict[str, Dict[str, str]]  # filename -> {sha256: ...}
    outputs: Dict[str, str]  # split -> file


# -----------------------------
# Core
# -----------------------------
def embed_one_csv(
    csv_path: Path,
    seq_col: str,
    model_name: str,
    device: str,
    precision: str,
    batch_size: int,
    pooling: str,
    max_length: Optional[int],
    oom_backoff: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column '{seq_col}' in {csv_path}")

    embedder = create_embedding(
        model_name=model_name,
        dataset=df,
        column_seq=seq_col,
        name_device=device,
        precision=precision,
        oom_backoff=oom_backoff,
    )

    layer_meta = try_set_last_layer(embedder)

    if max_length is None:
        embedder.run_process(batch_size=batch_size, pool=pooling)
    else:
        embedder.run_process(max_length=max_length, batch_size=batch_size, pool=pooling)

    emb = embedder.coded_dataset
    del embedder
    cleanup_memory()
    return emb, layer_meta


def process_seed_folder(
    seed_dir: Path,
    dataset_id: str,
    model_name: str,
    device: str,
    precision: str,
    batch_size: int,
    pooling: str,
    max_length: Optional[int],
    oom_backoff: bool,
    seq_col: str,
    export_format: str,
    dataset_manifest: dict,
) -> None:
    train_csv = seed_dir / "train.csv"
    val_csv = seed_dir / "val.csv"
    test_csv = seed_dir / "test.csv"

    missing = [p.name for p in (train_csv, val_csv, test_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split CSV(s) in {seed_dir}: {missing}. "
            "Re-run split generation with CSV export enabled or adapt the script to use idx files."
        )

    split_yaml = seed_dir / "split.yaml"
    split_id = seed_dir.name
    if split_yaml.exists():
        try:
            with open(split_yaml, "r", encoding="utf-8") as f:
                s = yaml.safe_load(f)
            split_id = s.get("split", {}).get("id", split_id)
        except Exception:
            pass

    model_short = short_model_name(model_name)
    out_dir = seed_dir / f"embedding_{model_short}_mean_last_bs{batch_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "train.csv": {"sha256": sha256_file(train_csv)},
        "val.csv": {"sha256": sha256_file(val_csv)},
        "test.csv": {"sha256": sha256_file(test_csv)},
    }

    outputs: Dict[str, str] = {}
    layer_sel = "default(last)"

    for split_name, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        emb_df, layer_meta = embed_one_csv(
            csv_path=csv_path,
            seq_col=seq_col,
            model_name=model_name,
            device=device,
            precision=precision,
            batch_size=batch_size,
            pooling=pooling,
            max_length=max_length,
            oom_backoff=oom_backoff,
        )
        layer_sel = str(layer_meta.get("layer_selection", layer_sel))

        out_file = out_dir / f"{split_name}.{export_format}"
        if export_format == "parquet":
            emb_df.to_parquet(out_file, index=False)
        elif export_format == "csv":
            emb_df.to_csv(out_file, index=False)
        else:
            raise ValueError(f"Unsupported export_format: {export_format}")

        outputs[split_name] = out_file.name
        del emb_df
        cleanup_memory()

    sylphy_ver = None
    if pkg_version is not None:
        try:
            sylphy_ver = pkg_version("sylphy")
        except Exception:
            sylphy_ver = None

    schema = EmbeddingSchema(
        mcs_version=str(dataset_manifest.get("mcs_version", "0.1")),
        created_utc=now_utc(),
        dataset_id=dataset_id,
        split_id=split_id,
        model_name=model_name,
        model_short=model_short,
        device=device,
        precision=precision,
        batch_size=batch_size,
        pooling=pooling,
        max_length=max_length,
        layer_selection=layer_sel,
        sylphy_version=sylphy_ver,
        inputs=inputs,
        outputs=outputs,
    )

    with open(out_dir / "schema.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(schema), f, sort_keys=False, allow_unicode=True)


def iter_seed_folders(splits_root: Path) -> List[Path]:
    seed_dirs: List[Path] = []
    if not splits_root.exists():
        return seed_dirs

    for strategy_dir in sorted([p for p in splits_root.iterdir() if p.is_dir()]):
        for seed_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
            seed_dirs.append(seed_dir)
    return seed_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to case_demo_01 (contains dataset/, schemas/, splits/)",
    )
    ap.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Optional override for schemas/dataset.yaml",
    )
    ap.add_argument(
        "--splits-root",
        type=Path,
        default=None,
        help="Optional override for splits/ root",
    )

    ap.add_argument("--model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    ap.add_argument("--precision", type=str, default="fp16", help="fp32, fp16, bf16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "eos"])
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--oom-backoff", action="store_true")
    ap.add_argument("--seq-col", type=str, default="sequence")
    ap.add_argument("--export-format", type=str, default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--limit", type=int, default=None, help="Optional limit number of seed folders (debug)")
    args = ap.parse_args()

    dataset_yaml = args.dataset_yaml or (args.case_dir / "schemas" / "dataset.yaml")
    splits_root = args.splits_root or (args.case_dir / "splits")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset_manifest = yaml.safe_load(f)

    dataset_id = dataset_manifest.get("dataset", {}).get("id", "unknown_dataset_id")

    seed_dirs = iter_seed_folders(splits_root)
    if args.limit is not None:
        seed_dirs = seed_dirs[: args.limit]

    if not seed_dirs:
        raise FileNotFoundError(f"No seed folders found under: {splits_root}")

    print(f"Found {len(seed_dirs)} seed folders under {splits_root}")
    print(f"dataset_id={dataset_id}")
    print(f"model={args.model_name} pooling={args.pooling} batch_size={args.batch_size} device={args.device} precision={args.precision}")

    for i, seed_dir in enumerate(seed_dirs, start=1):
        print(f"[{i}/{len(seed_dirs)}] Embedding: {seed_dir}")
        process_seed_folder(
            seed_dir=seed_dir,
            dataset_id=dataset_id,
            model_name=args.model_name,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            pooling=args.pooling,
            max_length=args.max_length,
            oom_backoff=args.oom_backoff,
            seq_col=args.seq_col,
            export_format=args.export_format,
            dataset_manifest=dataset_manifest,
        )
        cleanup_memory()

    print("Done ✅")


if __name__ == "__main__":
    main()
