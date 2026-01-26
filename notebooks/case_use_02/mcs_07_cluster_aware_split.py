#!/usr/bin/env python3
"""
MCS Case Study 02 — Script 7: Cluster-aware split generation (leave-cluster-out)

Goal
----
Generate split pools (train/val/test indices) where *clusters are disjoint* across splits.
This exposes structural fragility by preventing similarity / redundancy leakage.

Expected inputs (case-dir)
--------------------------
case_demo_01/
  dataset/amp_raw_10k.csv
  schemas/dataset.yaml
  similarity/
    embeddings.parquet          # output of mcs_06_similarity_graph.py
    similarity_schema.yaml

Outputs
-------
case_demo_01/
  clusters/
    labels.csv                  # dataset_index, sequence, cluster_id
    cluster_schema.yaml
  splits/cluster_aware/seed_<SEED>/
    train_idx.npy
    val_idx.npy
    test_idx.npy
    split.yaml                  # MCS-style split schema (cluster-aware)

Notes
-----
- Default clustering uses KMeans (sklearn), to avoid extra deps.
- Splits are made by assigning *whole clusters* to test/val/train until target sizes
  are reached (greedy packing on shuffled cluster order).
- If raw dataset has duplicated sequences, mapping sequence->dataset_index is ambiguous;
  we keep the first occurrence and warn.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# -----------------------------
# Schemas
# -----------------------------
@dataclass
class ClusterSchema:
    mcs_version: str
    created_utc: str
    method: str
    params: Dict[str, object]
    n_samples: int
    n_clusters: int
    artifacts: Dict[str, str]


@dataclass
class SplitSchema:
    mcs_version: str
    created_utc: str
    split: Dict[str, object]
    dataset: Dict[str, object]
    constraints: Dict[str, object]
    artifacts: Dict[str, str]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seeds(seeds: Optional[str], n_seeds: int, seed_start: int) -> List[int]:
    if seeds:
        s = seeds.strip()
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return list(range(seed_start, seed_start + n_seeds))


def infer_embedding_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p_")]
    if cols:
        try:
            cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
        except Exception:
            cols = sorted(cols)
    return cols


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def greedy_cluster_split(
    cluster_ids: List[int],
    cluster_sizes: Dict[int, int],
    rng: np.random.RandomState,
    n_total: int,
    test_size: float,
    val_size: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Assign whole clusters to (train, val, test) by greedy packing."""
    shuffled = cluster_ids.copy()
    rng.shuffle(shuffled)

    n_test_target = int(round(test_size * n_total))
    n_val_target = int(round(val_size * n_total))

    test_clusters: List[int] = []
    val_clusters: List[int] = []

    n_test = 0
    n_val = 0

    # fill test
    for cid in shuffled:
        if n_test < n_test_target:
            test_clusters.append(cid)
            n_test += cluster_sizes[cid]
        else:
            break

    remaining = [cid for cid in shuffled if cid not in set(test_clusters)]

    # fill val
    for cid in remaining:
        if n_val < n_val_target:
            val_clusters.append(cid)
            n_val += cluster_sizes[cid]
        else:
            break

    assigned = set(test_clusters) | set(val_clusters)
    train_clusters = [cid for cid in shuffled if cid not in assigned]

    return train_clusters, val_clusters, test_clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--embeddings-parquet", type=Path, default=None,
                    help="Default: <case-dir>/similarity/embeddings.parquet")
    ap.add_argument("--similarity-schema", type=Path, default=None,
                    help="Default: <case-dir>/similarity/similarity_schema.yaml")
    ap.add_argument("--raw-csv", type=Path, default=None,
                    help="Default: <case-dir>/dataset/amp_raw_10k.csv")
    ap.add_argument("--seq-col", type=str, default="sequence")
    ap.add_argument("--target-col", type=str, default="Antimicrobial")

    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans"])
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize embeddings before clustering (recommended for cosine geometry).")

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)

    ap.add_argument("--seeds", type=str, default=None,
                    help="Seeds list '1,2,3' or range '1-10'. If omitted, uses --n-seeds and --seed-start.")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1)

    ap.add_argument("--strategy-name", type=str, default="cluster_aware")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    case_dir = args.case_dir
    embeddings_parquet = args.embeddings_parquet or (case_dir / "similarity" / "embeddings.parquet")
    similarity_schema = args.similarity_schema or (case_dir / "similarity" / "similarity_schema.yaml")
    raw_csv = args.raw_csv or (case_dir / "dataset" / "amp_raw_10k.csv")

    if not embeddings_parquet.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {embeddings_parquet}")
    if not similarity_schema.exists():
        raise FileNotFoundError(f"Missing similarity schema: {similarity_schema}")
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_csv}")

    # Load dataset manifest for mcs_version + dataset_id
    dataset_yaml = case_dir / "schemas" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {dataset_yaml}")
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset_manifest = yaml.safe_load(f)
    mcs_version = str(dataset_manifest.get("mcs_version", "0.1"))
    dataset_id = dataset_manifest.get("dataset", {}).get("id", "unknown_dataset_id")

    # Raw dataset mapping: sequence -> first dataset_index
    raw = pd.read_csv(raw_csv)
    if args.seq_col not in raw.columns or args.target_col not in raw.columns:
        raise ValueError(f"Raw dataset must contain '{args.seq_col}' and '{args.target_col}'.")
    raw_seq = raw[args.seq_col].astype(str)

    if raw_seq.duplicated().any():
        ndup = int(raw_seq.duplicated().sum())
        print(f"[WARN] Raw dataset has {ndup} duplicated sequences. Using first occurrence for mapping.")

    seq_to_index: Dict[str, int] = {}
    for i, s in enumerate(raw_seq.to_list()):
        if s not in seq_to_index:
            seq_to_index[s] = int(i)

    # Load embeddings (unique sequences expected)
    emb = pd.read_parquet(embeddings_parquet)
    if args.seq_col not in emb.columns:
        raise ValueError(f"Embeddings must include '{args.seq_col}' column.")

    emb_seq = emb[args.seq_col].astype(str)
    emb_cols = infer_embedding_cols(emb)
    if not emb_cols:
        raise ValueError("No embedding columns found (p_*).")

    mapped_idx = emb_seq.map(lambda s: seq_to_index.get(s, None))
    if mapped_idx.isna().any():
        missing = int(mapped_idx.isna().sum())
        raise ValueError(f"Failed to map {missing} embedding sequences to raw dataset indices.")
    dataset_index = mapped_idx.astype(int).to_numpy()

    X = emb[emb_cols].to_numpy(dtype=np.float32)
    if args.normalize:
        X = normalize(X, norm="l2")

    # Clustering
    clusters_dir = case_dir / f"clusters_k{args.n_clusters}"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    labels_path = clusters_dir / "labels.csv"
    schema_path = clusters_dir / "cluster_schema.yaml"

    if labels_path.exists() and schema_path.exists() and not args.overwrite:
        print("[INFO] Cluster artifacts already exist. Reusing (use --overwrite to recompute).")
        labels_df = pd.read_csv(labels_path)
        if "cluster_id" not in labels_df.columns:
            raise ValueError("Existing labels.csv missing 'cluster_id'.")
    else:
        km = KMeans(n_clusters=args.n_clusters, random_state=0, n_init="auto")
        cluster_id = km.fit_predict(X).astype(int)

        labels_df = pd.DataFrame({
            "dataset_index": dataset_index,
            "sequence": emb_seq.to_numpy(),
            "cluster_id": cluster_id,
        }).drop_duplicates(subset=["dataset_index"]).sort_values("dataset_index").reset_index(drop=True)

        labels_df.to_csv(labels_path, index=False)

        c_schema = ClusterSchema(
            mcs_version=mcs_version,
            created_utc=now_utc(),
            method="kmeans",
            params={
                "n_clusters": int(args.n_clusters),
                "normalize": bool(args.normalize),
            },
            n_samples=int(labels_df.shape[0]),
            n_clusters=int(labels_df["cluster_id"].nunique()),
            artifacts={
                "labels": str(labels_path),
                "embeddings": str(embeddings_parquet),
                "similarity_schema": str(similarity_schema),
            },
        )
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(c_schema), f, sort_keys=False, allow_unicode=True)

        print(f"[OK] Wrote cluster labels: {labels_path}")
        print(f"[OK] Wrote cluster schema: {schema_path}")

    # Split generation (leave-cluster-out)
    split_root = case_dir / "splits" / args.strategy_name
    split_root.mkdir(parents=True, exist_ok=True)

    cluster_sizes = labels_df["cluster_id"].value_counts().to_dict()
    cluster_ids = sorted(cluster_sizes.keys())
    n_total = int(labels_df.shape[0])

    seeds = parse_seeds(args.seeds, args.n_seeds, args.seed_start)

    for seed in seeds:
        rng = np.random.RandomState(seed)
        train_c, val_c, test_c = greedy_cluster_split(
            cluster_ids=cluster_ids,
            cluster_sizes=cluster_sizes,
            rng=rng,
            n_total=n_total,
            test_size=args.test_size,
            val_size=args.val_size,
        )

        train_idx = labels_df[labels_df["cluster_id"].isin(train_c)]["dataset_index"].to_numpy(dtype=int)
        val_idx = labels_df[labels_df["cluster_id"].isin(val_c)]["dataset_index"].to_numpy(dtype=int)
        test_idx = labels_df[labels_df["cluster_id"].isin(test_c)]["dataset_index"].to_numpy(dtype=int)

        if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & set(test_idx)) or len(set(val_idx) & set(test_idx)):
            raise RuntimeError("Split indices overlap (should not happen).")

        seed_dir = split_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        np.save(seed_dir / "train_idx.npy", np.sort(train_idx))
        np.save(seed_dir / "val_idx.npy", np.sort(val_idx))
        np.save(seed_dir / "test_idx.npy", np.sort(test_idx))

        split_id = f"{args.strategy_name}_seed_{seed}"

        split_schema = SplitSchema(
            mcs_version=mcs_version,
            created_utc=now_utc(),
            split={
                "id": split_id,
                "strategy": args.strategy_name,
                "seed": int(seed),
                "sizes": {
                    "n_total": int(n_total),
                    "train": int(len(train_idx)),
                    "val": int(len(val_idx)),
                    "test": int(len(test_idx)),
                    "train_frac": float(len(train_idx) / n_total),
                    "val_frac": float(len(val_idx) / n_total),
                    "test_frac": float(len(test_idx) / n_total),
                },
                "clusters": {
                    "n_clusters_total": int(len(cluster_ids)),
                    "train_clusters": int(len(train_c)),
                    "val_clusters": int(len(val_c)),
                    "test_clusters": int(len(test_c)),
                },
            },
            dataset={
                "id": str(dataset_id),
                "raw_csv": str(raw_csv),
                "raw_sha256": sha256_file(raw_csv),
                "labels_source": str(labels_path),
                "labels_sha256": sha256_file(labels_path),
                "similarity_schema": str(similarity_schema),
                "similarity_schema_sha256": sha256_file(similarity_schema),
            },
            constraints={
                "cluster_disjoint_splits": True,
                "test_size_target": float(args.test_size),
                "val_size_target": float(args.val_size),
                "clustering_method": "kmeans",
                "n_clusters": int(args.n_clusters),
                "normalize": bool(args.normalize),
            },
            artifacts={
                "train_idx": "train_idx.npy",
                "val_idx": "val_idx.npy",
                "test_idx": "test_idx.npy",
                "split_yaml": "split.yaml",
            },
        )

        with open(seed_dir / "split.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(split_schema), f, sort_keys=False, allow_unicode=True)

        print(f"[OK] Split written: {seed_dir}  (train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")

    print("\nDone ✅")
    print("Next: run embeddings for these splits with mcs_03_embed_split_pools_v2.py, then train with mcs_04_train_models_split_pools_v1.py.")


if __name__ == "__main__":
    main()
