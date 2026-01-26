#!/usr/bin/env python3
"""
MCS Case Study — Script 3 (v5): Embed split pools with Sylphy (IDX/CSV-aware, robust import)

Fix
---
Your environment has `sylphy` but *not* `sylphy.pipeline`.
So this version does NOT assume a specific Sylphy module path.

Instead, it resolves `create_embedding` dynamically by:
  1) trying common import locations, then
  2) walking Sylphy submodules to find a callable named `create_embedding`.

It then uses your exact embedding pattern:
  embedder = create_embedding(...)
  layer_meta = try_set_last_layer(embedder)
  embedder.run_process(...)
  emb = embedder.coded_dataset

Supports BOTH split formats
---------------------------
A) CSV mode:
   seed_dir/train.csv, val.csv, test.csv
B) IDX mode:
   seed_dir/train_idx.npy, val_idx.npy, test_idx.npy  (+ raw dataset CSV)

Folder layout expected (case-dir)
---------------------------------
case_demo_01/
  dataset/amp_raw_10k.csv
  schemas/dataset.yaml
  splits/<strategy>/seed_<SEED>/
    train_idx.npy, val_idx.npy, test_idx.npy, split.yaml
    # OR train.csv,val.csv,test.csv (optional)
    embedding_<...>/  (created)

Outputs per seed
----------------
seed_<SEED>/embedding_<MODEL>_<POOL>_last_bs<B>_<DEV>_<PREC>/
  train.parquet
  val.parquet
  test.parquet
  schema.yaml
"""

from __future__ import annotations

import argparse
import gc
import importlib
import pkgutil
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# -----------------------------
# Schemas
# -----------------------------
@dataclass
class EmbeddingSchema:
    mcs_version: str
    created_utc: str
    run_id: str
    dataset_id: str
    split_id: str
    embedding: Dict[str, object]
    inputs: Dict[str, Dict[str, str]]
    outputs: Dict[str, str]
    environment: Dict[str, object]
    sylphy: Dict[str, object]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_freeze() -> Optional[List[str]]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        return out.decode().splitlines()
    except Exception:
        return None


def infer_seed_folders(case_dir: Path) -> List[Path]:
    return sorted([p for p in (case_dir / "splits").glob("**/seed_*") if p.is_dir()])


def ensure_seq_col(df: pd.DataFrame, seq_col: str, context: str) -> None:
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column '{seq_col}' in {context}")


# -----------------------------
# Sylphy resolver (robust)
# -----------------------------
def resolve_create_embedding() -> Tuple[object, str]:
    """
    Returns (callable, origin_string) for Sylphy's create_embedding.
    Tries common locations, otherwise walks submodules.
    """
    # 1) common guesses
    candidates = [
        ("sylphy", "create_embedding"),
        ("sylphy.embedding", "create_embedding"),
        ("sylphy.embeddings", "create_embedding"),
        ("sylphy.core", "create_embedding"),
        ("sylphy.api", "create_embedding"),
        ("sylphy.utils", "create_embedding"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn, f"{mod_name}.{attr}"
        except Exception:
            pass

    # 2) walk submodules
    try:
        import sylphy  # type: ignore
    except Exception as e:
        raise ImportError(f"'sylphy' is not importable in this env: {e}")

    if not hasattr(sylphy, "__path__"):
        # single-file module fallback
        fn = getattr(sylphy, "create_embedding", None)
        if callable(fn):
            return fn, "sylphy.create_embedding"
        raise ImportError("Could not resolve create_embedding: sylphy has no __path__ and no create_embedding attr.")

    for m in pkgutil.walk_packages(sylphy.__path__, prefix="sylphy."):
        name = m.name
        try:
            mod = importlib.import_module(name)
            fn = getattr(mod, "create_embedding", None)
            if callable(fn):
                return fn, f"{name}.create_embedding"
        except Exception:
            continue

    raise ImportError(
        "Could not resolve Sylphy create_embedding. "
        "Tip: run `python -c \"import sylphy; import pkgutil; print([m.name for m in pkgutil.walk_packages(sylphy.__path__, 'sylphy.')][:50])\"` "
        "to inspect submodules."
    )


def try_set_last_layer(embedder) -> Dict[str, object]:
    """
    Best-effort: set embedder to last layer if API supports it.
    Returns meta dict describing what happened.
    """
    meta: Dict[str, object] = {"requested": "last", "ok": False}

    # Try common attribute names
    for attr in ["layer", "layers", "layer_index", "layer_idx", "selected_layer", "repr_layer", "representation_layer"]:
        if hasattr(embedder, attr):
            try:
                setattr(embedder, attr, "last")
                meta.update({"ok": True, "method": f"setattr:{attr}", "value": "last"})
                return meta
            except Exception:
                pass

    # Try common methods
    for meth in ["set_last_layer", "select_last_layer", "use_last_layer", "set_layer", "select_layer"]:
        if hasattr(embedder, meth):
            try:
                fn = getattr(embedder, meth)
                try:
                    fn()
                    meta.update({"ok": True, "method": f"call:{meth}", "value": None})
                    return meta
                except TypeError:
                    fn("last")
                    meta.update({"ok": True, "method": f"call:{meth}", "value": "last"})
                    return meta
            except Exception:
                pass

    meta.update({"ok": False, "method": "not_supported"})
    return meta


def embed_one_df(
    df: pd.DataFrame,
    seq_col: str,
    create_embedding_fn,
    model_name: str,
    device: str,
    precision: str,
    batch_size: int,
    pooling: str,
    max_length: Optional[int],
    oom_backoff: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Implements exactly:
      embedder = create_embedding(...)
      layer_meta = try_set_last_layer(embedder)
      embedder.run_process(...)
      emb = embedder.coded_dataset
    """
    ensure_seq_col(df, seq_col, "input dataframe")

    embedder = create_embedding_fn(
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


def read_split_idx(seed_dir: Path, raw: pd.DataFrame, name: str) -> pd.DataFrame:
    p_npy = seed_dir / f"{name}_idx.npy"
    if not p_npy.exists():
        raise FileNotFoundError(p_npy)
    idx = np.load(p_npy).astype(int)
    out = raw.iloc[idx].copy()
    out = out.reset_index(drop=False).rename(columns={"index": "dataset_index"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--dataset-yaml", type=Path, default=None)
    ap.add_argument("--raw-csv", type=Path, default=None)

    ap.add_argument("--seq-col", type=str, default="sequence")
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--pooling", type=str, default="mean")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--oom-backoff", action="store_true")

    ap.add_argument("--force", action="store_true", help="Recompute embeddings even if folder exists.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    case_dir = args.case_dir
    dataset_yaml = args.dataset_yaml or (case_dir / "schemas" / "dataset.yaml")
    raw_csv = args.raw_csv or (case_dir / "dataset" / "amp_raw_10k.csv")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    mcs_version = str(manifest.get("mcs_version", "0.1"))
    dataset_id = manifest.get("dataset", {}).get("id", "unknown_dataset_id")

    raw = pd.read_csv(raw_csv)

    create_embedding_fn, origin = resolve_create_embedding()
    print(f"[INFO] Resolved create_embedding from: {origin}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_folders = infer_seed_folders(case_dir)
    if args.limit is not None:
        seed_folders = seed_folders[: args.limit]

    print(f"Found {len(seed_folders)} seed folders under {case_dir / 'splits'}")
    print(f"dataset_id={dataset_id}")
    print(f"model={args.model_name} pooling={args.pooling} batch_size={args.batch_size} device={args.device} precision={args.precision}")

    for i, seed_dir in enumerate(seed_folders, start=1):
        print(f"[{i}/{len(seed_folders)}] Embedding: {seed_dir}")

        split_yaml = seed_dir / "split.yaml"
        split_id = seed_dir.name
        if split_yaml.exists():
            with open(split_yaml, "r", encoding="utf-8") as f:
                s = yaml.safe_load(f)
            split_id = s.get("split", {}).get("id", split_id)

        embed_id = f"embedding_{args.model_name.split('/')[-1]}_{args.pooling}_last_bs{args.batch_size}_{args.device}_{args.precision}"
        out_dir = seed_dir / embed_id

        if out_dir.exists() and not args.force:
            if (out_dir / "train.parquet").exists() and (out_dir / "val.parquet").exists() and (out_dir / "test.parquet").exists():
                print(f"  [SKIP] {out_dir} already exists.")
                continue

        out_dir.mkdir(parents=True, exist_ok=True)

        def load_split(name: str) -> pd.DataFrame:
            p_csv = seed_dir / f"{name}.csv"
            p_npy = seed_dir / f"{name}_idx.npy"
            if p_csv.exists():
                df = pd.read_csv(p_csv)
                ensure_seq_col(df, args.seq_col, str(p_csv))
                return df
            if p_npy.exists():
                df = read_split_idx(seed_dir, raw, name)
                ensure_seq_col(df, args.seq_col, str(p_npy))
                return df
            raise FileNotFoundError(f"Missing both {name}.csv and {name}_idx.npy in {seed_dir}")

        train_df = load_split("train")
        val_df = load_split("val")
        test_df = load_split("test")

        layer_meta_all: Dict[str, object] = {}

        train_emb, layer_meta = embed_one_df(
            train_df, args.seq_col, create_embedding_fn,
            args.model_name, args.device, args.precision,
            args.batch_size, args.pooling, args.max_length, args.oom_backoff
        )
        layer_meta_all["train"] = layer_meta
        train_emb.to_parquet(out_dir / "train.parquet", index=False)

        val_emb, layer_meta = embed_one_df(
            val_df, args.seq_col, create_embedding_fn,
            args.model_name, args.device, args.precision,
            args.batch_size, args.pooling, args.max_length, args.oom_backoff
        )
        layer_meta_all["val"] = layer_meta
        val_emb.to_parquet(out_dir / "val.parquet", index=False)

        test_emb, layer_meta = embed_one_df(
            test_df, args.seq_col, create_embedding_fn,
            args.model_name, args.device, args.precision,
            args.batch_size, args.pooling, args.max_length, args.oom_backoff
        )
        layer_meta_all["test"] = layer_meta
        test_emb.to_parquet(out_dir / "test.parquet", index=False)

        # Feature dim (best-effort)
        p_cols = [c for c in train_emb.columns if isinstance(c, str) and c.startswith("p_")]
        feature_dim = int(len(p_cols)) if p_cols else None

        schema = EmbeddingSchema(
            mcs_version=mcs_version,
            created_utc=now_utc(),
            run_id=run_id,
            dataset_id=str(dataset_id),
            split_id=str(split_id),
            embedding={
                "model_name": args.model_name,
                "pooling": args.pooling,
                "layer": "last",
                "batch_size": int(args.batch_size),
                "device": args.device,
                "precision": args.precision,
                "max_length": int(args.max_length) if args.max_length is not None else None,
                "feature_dim": feature_dim,
                "create_embedding_origin": origin,
            },
            inputs={
                "dataset_yaml": {"path": str(dataset_yaml), "sha256": sha256_file(dataset_yaml)},
                "raw_csv": {"path": str(raw_csv), "sha256": sha256_file(raw_csv)},
                "split_yaml": {"path": str(split_yaml), "sha256": sha256_file(split_yaml)} if split_yaml.exists() else {"path": str(split_yaml), "sha256": ""},
            },
            outputs={
                "train": "train.parquet",
                "val": "val.parquet",
                "test": "test.parquet",
                "schema": "schema.yaml",
            },
            environment={
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "packages": pip_freeze(),
            },
            sylphy={
                "layer_meta": layer_meta_all,
            },
        )

        with open(out_dir / "schema.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(schema), f, sort_keys=False, allow_unicode=True)

        cleanup_memory()
        print(f"  [OK] wrote train/val/test parquet in {out_dir}")

    print("Done ✅")


if __name__ == "__main__":
    main()
