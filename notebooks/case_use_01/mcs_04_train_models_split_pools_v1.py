#!/usr/bin/env python3
"""
MCS Case Study — Script 4 (generalized): Train classical models over embedding split pools + schemas + predictions

Supports:
- RandomForest (rf)
- SVM (svm)  [SVC with probability=True]
- kNN (knn)
- Logistic Regression (lr)

Assumes embedding artifacts contain:
- sequence column (default: "sequence")
- embedding columns p_0, p_1, ... (or columns starting with "p_")
and DO NOT contain labels.

Labels are reconstructed from:
- raw dataset: dataset/amp_raw_10k.csv
- split indices: train_idx.npy, val_idx.npy, test_idx.npy

Exports per (seed, embedding, model):
seed_<SEED>/models/<model_name>/<embedding_id>/
  model.joblib
  metrics.json
  training_schema.yaml
  preds_train.csv
  preds_val.csv
  preds_test.csv

Run-level provenance:
case_dir/executions/<RUN_ID>/execution.yaml
case_dir/executions/<RUN_ID>/summary.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# -----------------------------
# Utils
# -----------------------------
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


def git_info(case_dir: Path) -> Dict[str, Optional[str]]:
    def run(cmd: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(cmd, cwd=case_dir, stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": run(["git", "status", "--porcelain"]),
    }


def pip_freeze() -> Optional[List[str]]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        return out.decode().splitlines()
    except Exception:
        return None


# -----------------------------
# Schemas
# -----------------------------
@dataclass
class ExecutionSchema:
    mcs_version: str
    created_utc: str
    run_id: str
    case_dir: str
    dataset_id: str
    command: str
    python: str
    platform: str
    git: Dict[str, Optional[str]]
    packages: Optional[List[str]]
    notes: str


@dataclass
class TrainingSchema:
    mcs_version: str
    created_utc: str
    run_id: str
    dataset_id: str
    split_id: str
    embedding_id: str
    embedding_schema_path: str
    model: Dict[str, Any]
    inputs: Dict[str, Dict[str, str]]
    outputs: Dict[str, str]
    metrics: Dict[str, float]
    alignment: Dict[str, object]


# -----------------------------
# Loading helpers
# -----------------------------
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_embedding_feature_cols(df: pd.DataFrame) -> List[str]:
    p_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p_")]
    if p_cols:
        try:
            p_cols = sorted(p_cols, key=lambda x: int(x.split("_")[1]))
        except Exception:
            p_cols = sorted(p_cols)
        return p_cols
    # fallback: numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def load_X_seq(path: Path, seq_col: str) -> Tuple[np.ndarray, pd.Series, List[str]]:
    df = read_table(path)
    seq = df[seq_col].astype(str) if seq_col in df.columns else pd.Series(["<missing_seq>"] * len(df))
    feat_cols = infer_embedding_feature_cols(df)
    if not feat_cols:
        raise ValueError(f"No embedding feature columns found in {path}.")
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, seq, feat_cols


def y_from_indices(raw_y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return raw_y[idx].astype(int)


def validate_or_map_labels(
    split_name: str,
    emb_seq: pd.Series,
    raw_seq_split: pd.Series,
    y_idx: np.ndarray,
    raw_seq_to_y: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Primary: use idx-derived labels (y_idx), but verify order by comparing sequences.
    Fallback: map by sequence (requires sequences exist in raw dataset; duplicates can be ambiguous).
    """
    alignment: Dict[str, object] = {"method": "idx", "ok": True, "n": int(len(emb_seq))}
    if len(emb_seq) != len(raw_seq_split):
        alignment.update({"ok": False, "reason": f"len_mismatch emb={len(emb_seq)} raw={len(raw_seq_split)}"})
        y_map = emb_seq.map(lambda s: raw_seq_to_y.get(str(s), None))
        if y_map.isna().any():
            missing = int(y_map.isna().sum())
            raise ValueError(f"[{split_name}] Sequence mapping failed for {missing} sequences.")
        alignment.update({"method": "sequence_map", "ok": True, "reason": "len_mismatch_fallback"})
        return y_map.astype(int).to_numpy(), alignment

    if (emb_seq.astype(str).to_numpy() == raw_seq_split.astype(str).to_numpy()).all():
        return y_idx, alignment

    alignment.update({"ok": False, "reason": "order_mismatch_fallback"})
    y_map = emb_seq.map(lambda s: raw_seq_to_y.get(str(s), None))
    if y_map.isna().any():
        missing = int(y_map.isna().sum())
        raise ValueError(f"[{split_name}] Sequence mapping failed for {missing} sequences.")
    alignment.update({"method": "sequence_map", "ok": True})
    return y_map.astype(int).to_numpy(), alignment


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auprc"] = float("nan")
    return out


# -----------------------------
# Model factory + selection
# -----------------------------
def model_default_grid(model: str) -> Dict[str, List[Any]]:
    """
    ParameterGrid is applied to the *estimator step* name, depending on model.
    We build pipelines for svm/knn/lr with 'scaler' + 'clf'.
    For rf: direct estimator (no scaler needed).
    """
    if model == "rf":
        return {
            "n_estimators": [500],
            "max_depth": [None],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "class_weight": ["balanced"],
        }
    if model == "svm":
        # SVC can be expensive; keep it small by default
        return {
            "clf__C": [1.0],
            "clf__kernel": ["rbf"],
            "clf__gamma": ["scale"],
        }
    if model == "knn":
        return {
            "clf__n_neighbors": [5, 15],
            "clf__weights": ["uniform", "distance"],
        }
    if model == "lr":
        return {
            "clf__C": [1.0, 0.3, 3.0],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [2000],
        }
    raise ValueError(f"Unknown model: {model}")


def model_grid_search_grid(model: str) -> Dict[str, List[Any]]:
    if model == "rf":
        return {
            "n_estimators": [200, 500],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        }
    if model == "svm":
        return {
            "clf__C": [0.3, 1.0, 3.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale"],
        }
    if model == "knn":
        return {
            "clf__n_neighbors": [3, 5, 11, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # Manhattan vs Euclidean
        }
    if model == "lr":
        return {
            "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [3000],
        }
    raise ValueError(f"Unknown model: {model}")


def build_estimator(model: str, seed: int, n_jobs: int) -> Any:
    if model == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if model == "svm":
        # probability=True enables predict_proba; can be slower but simplest
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=seed)),
        ])
    if model == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ])
    if model == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed)),
        ])
    raise ValueError(f"Unknown model: {model}")


def get_probabilities(estimator: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    # fallback: decision_function -> sigmoid-ish scaling (not ideal). For our supported models, predict_proba exists.
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        # logistic transform
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Estimator does not support predict_proba or decision_function.")


def set_params_compat(estimator: Any, params: Dict[str, Any]) -> Any:
    estimator.set_params(**params)
    return estimator


def fit_best_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    param_grid: Dict[str, List[Any]],
    selection_metric: str,
    n_jobs: int,
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    best_params: Dict[str, Any] = {}
    best_score = -1.0
    best_val_metrics: Dict[str, float] = {}

    for params in ParameterGrid(param_grid):
        est = build_estimator(model, seed=seed, n_jobs=n_jobs)
        set_params_compat(est, params)
        est.fit(X_train, y_train)

        y_prob = get_probabilities(est, X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(y_val, y_prob, y_pred)
        score = float(m.get(selection_metric, float("nan")))
        if np.isnan(score):
            score = -1.0

        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_val_metrics = dict(m)

        del est
        cleanup_memory()

    if not best_params:
        raise RuntimeError("Failed to select hyperparameters (empty best_params).")

    # Refit on train+val
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    final = build_estimator(model, seed=seed, n_jobs=n_jobs)
    set_params_compat(final, best_params)
    final.fit(X_tv, y_tv)

    cleanup_memory()
    return final, best_params, best_val_metrics


# -----------------------------
# IO helpers
# -----------------------------
def find_embedding_dirs(case_dir: Path) -> List[Path]:
    return sorted([p for p in (case_dir / "splits").glob("**/seed_*/embedding_*") if p.is_dir()])


def write_preds(
    out_path: Path,
    idx: np.ndarray,
    seq: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    dfp = pd.DataFrame({
        "dataset_index": idx.astype(int),
        "sequence": seq.astype(str).to_numpy(),
        "y_true": y_true.astype(int),
        "y_prob": y_prob.astype(float),
        "y_pred": y_pred.astype(int),
    })
    dfp.to_csv(out_path, index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True, choices=["rf", "svm", "knn", "lr"],
                    help="Which algorithm to train: rf | svm | knn | lr")
    ap.add_argument("--dataset-yaml", type=Path, default=None)
    ap.add_argument("--raw-csv", type=Path, default=None)
    ap.add_argument("--seq-col", type=str, default="sequence")
    ap.add_argument("--target-col", type=str, default="Antimicrobial")
    ap.add_argument("--selection-metric", type=str, default="auprc", choices=["auprc", "auroc", "f1"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--grid", action="store_true", help="Enable a broader hyperparameter grid (still reasonable).")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of embedding folders (debug).")
    args = ap.parse_args()

    case_dir = args.case_dir
    dataset_yaml = args.dataset_yaml or (case_dir / "schemas" / "dataset.yaml")
    raw_csv = args.raw_csv or (case_dir / "dataset" / "amp_raw_10k.csv")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset_manifest = yaml.safe_load(f)
    dataset_id = dataset_manifest.get("dataset", {}).get("id", "unknown_dataset_id")
    mcs_version = str(dataset_manifest.get("mcs_version", "0.1"))

    # execution.yaml immediately
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exec_dir = case_dir / "executions" / run_id
    exec_dir.mkdir(parents=True, exist_ok=True)

    exec_schema = ExecutionSchema(
        mcs_version=mcs_version,
        created_utc=now_utc(),
        run_id=run_id,
        case_dir=str(case_dir),
        dataset_id=str(dataset_id),
        command=" ".join([sys.executable] + sys.argv),
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        git=git_info(case_dir),
        packages=pip_freeze(),
        notes=f"Training over embedding split pools. Model={args.model}. Predictions exported for fragility/calibration analysis.",
    )
    with open(exec_dir / "execution.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(exec_schema), f, sort_keys=False, allow_unicode=True)

    # raw labels
    raw = pd.read_csv(raw_csv)
    if args.seq_col not in raw.columns or args.target_col not in raw.columns:
        raise ValueError(f"Raw dataset must contain '{args.seq_col}' and '{args.target_col}'.")

    raw_seq = raw[args.seq_col].astype(str)
    raw_y = pd.to_numeric(raw[args.target_col], errors="coerce").astype("Int64")
    if raw_y.isna().any():
        raise ValueError("NaNs in raw target column.")
    raw_y = raw_y.astype(int).to_numpy()

    if raw_seq.duplicated().any():
        ndup = int(raw_seq.duplicated().sum())
        print(f"[WARN] Raw dataset has {ndup} duplicated sequences. Sequence mapping may be ambiguous.")
    raw_seq_to_y = dict(zip(raw_seq.to_list(), raw_y.tolist()))

    emb_dirs = find_embedding_dirs(case_dir)
    if args.limit is not None:
        emb_dirs = emb_dirs[: args.limit]
    if not emb_dirs:
        raise FileNotFoundError("No embedding directories found under splits/.")

    param_grid = model_grid_search_grid(args.model) if args.grid else model_default_grid(args.model)

    summary_rows: List[Dict[str, object]] = []

    for i, emb_dir in enumerate(emb_dirs, start=1):
        print(f"[{i}/{len(emb_dirs)}] Training model={args.model} for: {emb_dir}")

        seed_dir = emb_dir.parent
        split_yaml = seed_dir / "split.yaml"
        split_id = seed_dir.name
        split_seed = None
        if split_yaml.exists():
            with open(split_yaml, "r", encoding="utf-8") as f:
                s = yaml.safe_load(f)
            split_id = s.get("split", {}).get("id", split_id)
            split_seed = s.get("split", {}).get("seed", None)

        train_idx = np.load(seed_dir / "train_idx.npy")
        val_idx = np.load(seed_dir / "val_idx.npy")
        test_idx = np.load(seed_dir / "test_idx.npy")

        y_train_idx = y_from_indices(raw_y, train_idx)
        y_val_idx = y_from_indices(raw_y, val_idx)
        y_test_idx = y_from_indices(raw_y, test_idx)

        raw_train_seq = raw_seq.iloc[train_idx].reset_index(drop=True)
        raw_val_seq = raw_seq.iloc[val_idx].reset_index(drop=True)
        raw_test_seq = raw_seq.iloc[test_idx].reset_index(drop=True)

        embedding_id = emb_dir.name
        embedding_schema_path = str((emb_dir / "schema.yaml").resolve())

        def pick_path(name: str) -> Path:
            p = emb_dir / f"{name}.parquet"
            if p.exists():
                return p
            p = emb_dir / f"{name}.csv"
            if p.exists():
                return p
            raise FileNotFoundError(f"Missing {name}.parquet or {name}.csv in {emb_dir}")

        train_path = pick_path("train")
        val_path = pick_path("val")
        test_path = pick_path("test")

        X_train, emb_train_seq, feat_cols = load_X_seq(train_path, args.seq_col)
        X_val, emb_val_seq, _ = load_X_seq(val_path, args.seq_col)
        X_test, emb_test_seq, _ = load_X_seq(test_path, args.seq_col)

        y_train, align_train = validate_or_map_labels("train", emb_train_seq, raw_train_seq, y_train_idx, raw_seq_to_y)
        y_val, align_val = validate_or_map_labels("val", emb_val_seq, raw_val_seq, y_val_idx, raw_seq_to_y)
        y_test, align_test = validate_or_map_labels("test", emb_test_seq, raw_test_seq, y_test_idx, raw_seq_to_y)

        if len(X_train) != len(y_train) or len(X_val) != len(y_val) or len(X_test) != len(y_test):
            raise ValueError("X/y length mismatch after alignment.")

        if split_seed is None:
            try:
                split_seed = int(seed_dir.name.replace("seed_", ""))
            except Exception:
                split_seed = 42
        split_seed = int(split_seed)

        estimator, best_params, best_val_metrics = fit_best_model(
            model=args.model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            seed=split_seed,
            param_grid=param_grid,
            selection_metric=args.selection_metric,
            n_jobs=args.n_jobs,
        )

        prob_train = get_probabilities(estimator, X_train)
        prob_val = get_probabilities(estimator, X_val)
        prob_test = get_probabilities(estimator, X_test)

        pred_train = (prob_train >= args.threshold).astype(int)
        pred_val = (prob_val >= args.threshold).astype(int)
        pred_test = (prob_test >= args.threshold).astype(int)

        m_train = compute_metrics(y_train, prob_train, pred_train)
        m_val = compute_metrics(y_val, prob_val, pred_val)
        m_test = compute_metrics(y_test, prob_test, pred_test)

        model_name = {"rf": "random_forest", "svm": "svm", "knn": "knn", "lr": "logistic_regression"}[args.model]
        out_dir = seed_dir / "models" / model_name / embedding_id
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "model.joblib"
        joblib.dump(estimator, model_path)

        metrics_obj = {
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "selection_metric": args.selection_metric,
            "threshold": args.threshold,
            "best_val_metrics_during_search": best_val_metrics,
            "best_params": best_params,
            "model": args.model,
        }
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, indent=2)

        preds_train_path = out_dir / "preds_train.csv"
        preds_val_path = out_dir / "preds_val.csv"
        preds_test_path = out_dir / "preds_test.csv"

        write_preds(preds_train_path, train_idx, emb_train_seq.reset_index(drop=True), y_train, prob_train, threshold=args.threshold)
        write_preds(preds_val_path, val_idx, emb_val_seq.reset_index(drop=True), y_val, prob_val, threshold=args.threshold)
        write_preds(preds_test_path, test_idx, emb_test_seq.reset_index(drop=True), y_test, prob_test, threshold=args.threshold)

        inputs = {
            "dataset_raw": {"path": str(raw_csv), "sha256": sha256_file(raw_csv)},
            "split_train_idx": {"path": str((seed_dir / "train_idx.npy")), "sha256": sha256_file(seed_dir / "train_idx.npy")},
            "split_val_idx": {"path": str((seed_dir / "val_idx.npy")), "sha256": sha256_file(seed_dir / "val_idx.npy")},
            "split_test_idx": {"path": str((seed_dir / "test_idx.npy")), "sha256": sha256_file(seed_dir / "test_idx.npy")},
            "emb_train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "emb_val": {"path": str(val_path), "sha256": sha256_file(val_path)},
            "emb_test": {"path": str(test_path), "sha256": sha256_file(test_path)},
        }
        outputs = {
            "model": "model.joblib",
            "metrics": "metrics.json",
            "preds_train": "preds_train.csv",
            "preds_val": "preds_val.csv",
            "preds_test": "preds_test.csv",
            "schema": "training_schema.yaml",
        }
        align = {"train": align_train, "val": align_val, "test": align_test}

        train_schema = TrainingSchema(
            mcs_version=mcs_version,
            created_utc=now_utc(),
            run_id=run_id,
            dataset_id=str(dataset_id),
            split_id=str(split_id),
            embedding_id=str(embedding_id),
            embedding_schema_path=embedding_schema_path,
            model={
                "name": model_name,
                "type": type(estimator).__name__,
                "best_params": best_params,
                "feature_dim": int(X_train.shape[1]),
                "feature_columns": "p_*" if any(c.startswith("p_") for c in feat_cols) else "numeric_fallback",
                "threshold": float(args.threshold),
                "selection_metric": args.selection_metric,
            },
            inputs=inputs,
            outputs=outputs,
            metrics={
                "train_auprc": float(m_train.get("auprc", np.nan)),
                "val_auprc": float(m_val.get("auprc", np.nan)),
                "test_auprc": float(m_test.get("auprc", np.nan)),
                "train_auroc": float(m_train.get("auroc", np.nan)),
                "val_auroc": float(m_val.get("auroc", np.nan)),
                "test_auroc": float(m_test.get("auroc", np.nan)),
                "test_f1": float(m_test.get("f1", np.nan)),
                "test_mcc": float(m_test.get("mcc", np.nan)),
                "test_balanced_accuracy": float(m_test.get("balanced_accuracy", np.nan)),
            },
            alignment=align,
        )

        schema_path = out_dir / "training_schema.yaml"
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(train_schema), f, sort_keys=False, allow_unicode=True)

        try:
            strategy = emb_dir.parents[2].name
        except Exception:
            strategy = "unknown"

        summary_rows.append({
            "run_id": run_id,
            "dataset_id": dataset_id,
            "strategy": strategy,
            "seed": split_seed,
            "split_id": split_id,
            "embedding_id": embedding_id,
            "model": model_name,
            "val_auprc": m_val.get("auprc", np.nan),
            "test_auprc": m_test.get("auprc", np.nan),
            "val_auroc": m_val.get("auroc", np.nan),
            "test_auroc": m_test.get("auroc", np.nan),
            "test_f1": m_test.get("f1", np.nan),
            "test_mcc": m_test.get("mcc", np.nan),
            "schema_path": str(schema_path.resolve()),
            "model_path": str(model_path.resolve()),
            "preds_test_path": str(preds_test_path.resolve()),
            "align_train_method": align_train.get("method"),
            "align_val_method": align_val.get("method"),
            "align_test_method": align_test.get("method"),
        })

        del estimator, X_train, X_val, X_test
        del y_train, y_val, y_test
        cleanup_memory()

    summary = pd.DataFrame(summary_rows).sort_values(["strategy", "seed"]).reset_index(drop=True)
    summary_path = exec_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Wrote:", summary_path)
    print("Execution:", exec_dir / "execution.yaml")
    print("Done")


if __name__ == "__main__":
    main()
