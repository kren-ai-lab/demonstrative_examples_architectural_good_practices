#!/usr/bin/env python3
"""
MCS Case Study 02 — Script 6
Similarity graph construction from protein embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Schema
# -----------------------------
@dataclass
class SimilaritySchema:
    mcs_version: str
    created_utc: str
    case: str
    embedding_source: str
    distance_metric: str
    k_neighbors: int
    normalization: str
    n_samples: int
    artifacts: dict


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_embedding_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("p_")]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", type=str, default="cosine")
    args = ap.parse_args()

    case_dir = args.case_dir
    out_dir = case_dir / "similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect ALL embeddings (train+val+test, all seeds, but unique rows)
    emb_files = sorted(case_dir.glob("splits/**/embedding_*/train.parquet")) + \
                sorted(case_dir.glob("splits/**/embedding_*/val.parquet")) + \
                sorted(case_dir.glob("splits/**/embedding_*/test.parquet"))

    if not emb_files:
        raise RuntimeError("No embedding parquet files found.")

    dfs = []
    for p in emb_files:
        df = pd.read_parquet(p)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sequence"])
    full.reset_index(drop=True, inplace=True)

    emb_cols = infer_embedding_cols(full)
    if not emb_cols:
        raise RuntimeError("No embedding columns found (p_*).")

    X = full[emb_cols].to_numpy(dtype=np.float32)
    X = normalize(X, norm="l2")

    # kNN graph
    nn = NearestNeighbors(
        n_neighbors=args.k + 1,
        metric=args.metric,
        n_jobs=-1,
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # remove self-neighbor (first column)
    knn_idx = indices[:, 1:]
    knn_dist = distances[:, 1:]

    # Save artifacts
    emb_out = out_dir / "embeddings.parquet"
    full.to_parquet(emb_out, index=False)

    knn_df = pd.DataFrame({
        "source_index": np.repeat(np.arange(len(full)), args.k),
        "neighbor_index": knn_idx.flatten(),
        "distance": knn_dist.flatten(),
    })
    knn_out = out_dir / "knn_graph.parquet"
    knn_df.to_parquet(knn_out, index=False)

    # Schema
    schema = SimilaritySchema(
        mcs_version="0.1",
        created_utc=now_utc(),
        case="case_study_02",
        embedding_source="Sylphy / ESM2 / mean pooling / last layer",
        distance_metric=args.metric,
        k_neighbors=args.k,
        normalization="l2",
        n_samples=len(full),
        artifacts={
            "embeddings": str(emb_out),
            "knn_graph": str(knn_out),
        },
    )

    with open(out_dir / "similarity_schema.yaml", "w") as f:
        yaml.safe_dump(asdict(schema), f, sort_keys=False)

    print("✔ Similarity graph created")
    print("✔ Samples:", len(full))
    print("✔ kNN edges:", len(knn_df))


if __name__ == "__main__":
    main()
