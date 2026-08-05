#!/usr/bin/env python3

"""Evaluate fixed model variants and select one representative variant per family.

The workflow is intentionally split into two methodological stages.

GRID stage
----------
Each predeclared model variant is fitted on TRAIN and evaluated on VALIDATION
for the reference RANDOM regime. TEST is never loaded during this stage.

SELECTION stage
---------------
One representative variant is selected per task and model family using the
median validation average precision across all reference-regime seeds.
The same selected variant is then frozen for every similarity regime.

FINAL stage
-----------
The frozen representative variant is fitted on TRAIN and evaluated once on
TEST for every task, regime, and partition seed. No per-split hyperparameter
selection and no threshold optimization are performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SCRIPT_VERSION = "2.1.0-fixed-grid-expanded"

GRID_RESULT_COLUMNS = [
    "task_id",
    "reference_regime",
    "seed",
    "split_id",
    "model_id",
    "variant_id",
    "parameters_json",
    "status",
    "score_type",
    "validation_n",
    "validation_positive_count",
    "validation_negative_count",
    "validation_prevalence",
    "validation_average_precision",
    "validation_roc_auc",
    "validation_threshold",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_mcc",
    "validation_f1",
    "validation_precision",
    "validation_recall_sensitivity",
    "validation_specificity",
    "validation_brier_score",
    "fit_seconds",
    "validation_score_seconds",
    "warning_count",
    "warnings",
    "error",
]

SELECTED_VARIANT_COLUMNS = [
    "task_id",
    "model_id",
    "variant_id",
    "parameters_json",
    "reference_regime",
    "selection_metric",
    "selection_seed_count",
    "validation_ap_median",
    "validation_ap_q1",
    "validation_ap_q3",
    "validation_ap_iqr",
    "validation_ap_mean",
    "validation_ap_standard_deviation",
    "validation_ap_minimum",
    "validation_ap_maximum",
    "selection_rank",
    "selected",
]

PERFORMANCE_COLUMNS = [
    "task_id",
    "regime_id",
    "seed",
    "split_id",
    "model_id",
    "variant_id",
    "parameters_json",
    "partition",
    "score_type",
    "threshold",
    "n",
    "positive_count",
    "negative_count",
    "prevalence",
    "average_precision",
    "roc_auc",
    "prediction_positive_fraction",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "f1",
    "precision",
    "recall_sensitivity",
    "specificity",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "brier_score",
    "run_manifest_path",
]

FAILURE_COLUMNS = [
    "stage",
    "task_id",
    "regime_id",
    "seed",
    "split_id",
    "model_id",
    "variant_id",
    "error_type",
    "error",
    "failure_path",
]


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def make_json_safe(value: Any) -> Any:
    """Convert values into JSON-safe objects."""
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def canonical_json(value: Any) -> str:
    """Serialize an object deterministically."""
    return json.dumps(
        make_json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write deterministic indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_safe(data),
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")

    return data


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return data


def package_version(name: str) -> str:
    """Return an installed package version."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Return an empty table that still preserves its CSV header."""
    return pd.DataFrame(columns=columns)


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate methodological and structural configuration fields."""
    required_sections = (
        "input",
        "dataset",
        "representation",
        "selection",
        "training",
        "evaluation",
        "models",
        "execution",
        "output",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    dataset = config["dataset"]

    for field in (
        "sequence_id_column",
        "label_column",
        "partition_column",
        "included_tasks",
        "included_regimes",
        "expected_partitions",
        "expected_seeds_per_task_regime",
    ):
        if field not in dataset:
            raise ValueError(
                f"Missing dataset configuration field: {field}"
            )

    if set(dataset["expected_partitions"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "expected_partitions must contain train, validation, and test."
        )

    selection = config["selection"]

    if selection.get("reference_regime") not in dataset[
        "included_regimes"
    ]:
        raise ValueError(
            "selection.reference_regime must be an included regime."
        )

    if selection.get("metric") != "average_precision":
        raise ValueError(
            "This implementation selects representative variants using "
            "validation average precision."
        )

    if selection.get("aggregation") != "median":
        raise ValueError(
            "selection.aggregation must be 'median'."
        )

    if config["training"].get(
        "refit_on_train_validation", False
    ):
        raise ValueError(
            "refit_on_train_validation must remain false."
        )

    supported_estimators = {
        "knn",
        "logistic_regression",
        "random_forest",
        "linear_svm",
    }
    enabled = 0

    for model_id, model_config in config["models"].items():
        if not model_config.get("enabled", True):
            continue

        enabled += 1
        estimator = model_config.get("estimator")

        if estimator not in supported_estimators:
            raise ValueError(
                f"Unsupported estimator '{estimator}' for '{model_id}'."
            )

        variants = model_config.get("variants")

        if not isinstance(variants, list) or not variants:
            raise ValueError(
                f"Model '{model_id}' must define explicit variants."
            )

        observed_ids: set[str] = set()

        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError(
                    f"Every variant for '{model_id}' must be a mapping."
                )

            variant_id = str(variant.get("id", "")).strip()

            if not variant_id:
                raise ValueError(
                    f"A variant for '{model_id}' lacks an ID."
                )

            if variant_id in observed_ids:
                raise ValueError(
                    f"Duplicated variant ID for '{model_id}': {variant_id}"
                )

            observed_ids.add(variant_id)

            if not isinstance(variant.get("parameters", {}), dict):
                raise ValueError(
                    f"Variant '{variant_id}' parameters must be a mapping."
                )

        if estimator in {
            "knn",
            "logistic_regression",
            "linear_svm",
        } and model_config.get("preprocessing") != "standard_scaler":
            raise ValueError(
                f"Model '{model_id}' must use StandardScaler."
            )

        if estimator == "random_forest" and model_config.get(
            "preprocessing"
        ) != "none":
            raise ValueError(
                "Random Forest must not use scaling."
            )

        if estimator == "linear_svm":
            if model_config.get("calibration") != "none":
                raise ValueError(
                    "Linear SVM calibration must remain disabled."
                )

            if model_config.get("score_type") != "decision_function":
                raise ValueError(
                    "Linear SVM must use decision_function scores."
                )

    if enabled == 0:
        raise ValueError("No enabled models were found.")


def verify_software(config: dict[str, Any]) -> None:
    """Optionally enforce an exact scikit-learn version."""
    software = config.get("software", {}).get("scikit_learn", {})
    expected = software.get("expected_version")
    enforce = bool(software.get("enforce_version", False))
    observed = package_version("scikit-learn")

    if enforce and expected and observed != str(expected):
        raise RuntimeError(
            "Unexpected scikit-learn version. "
            f"Expected {expected}, found {observed}."
        )


def verify_input_manifests(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate representation and partition provenance."""
    inputs = config["input"]
    representation_config = config["representation"]

    paths = {
        "representation_manifest": Path(
            inputs["representation_manifest"]
        ),
        "partition_manifest": Path(
            inputs["partition_generation_manifest"]
        ),
        "embeddings": Path(inputs["embeddings"]),
        "partition_index": Path(inputs["partition_index"]),
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    representation_manifest = load_json(
        paths["representation_manifest"]
    )
    partition_manifest = load_json(paths["partition_manifest"])

    if representation_config.get("verify_embedding_sha256", True):
        expected = representation_manifest.get("output", {}).get(
            "sha256"
        )
        observed = sha256_file(paths["embeddings"])

        if expected != observed:
            raise ValueError(
                "Embedding SHA-256 does not match its manifest."
            )

    if representation_config.get(
        "verify_partition_index_sha256", True
    ):
        expected = (
            partition_manifest.get("outputs", {})
            .get("partition_index", {})
            .get("sha256")
        )
        observed = sha256_file(paths["partition_index"])

        if expected != observed:
            raise ValueError(
                "Partition-index SHA-256 does not match its manifest."
            )

    if representation_config.get(
        "verify_shared_task_dataset_sha256", True
    ):
        representation_input_hash = representation_manifest.get(
            "input", {}
        ).get("sha256")
        partition_input_hash = partition_manifest.get(
            "task_dataset", {}
        ).get("sha256")

        if representation_input_hash != partition_input_hash:
            raise ValueError(
                "Representations and partitions were derived from "
                "different task datasets."
            )

    expected_dimension = int(
        representation_config["expected_dimension"]
    )
    observed_dimension = int(
        representation_manifest.get("representation", {}).get(
            "output_dimension", -1
        )
    )

    if observed_dimension != expected_dimension:
        raise ValueError(
            "Representation dimension differs from the configuration."
        )

    required_validations = (
        "constant_dimensionality",
        "input_output_identifier_correspondence",
        "no_infinite_values",
        "no_nan",
        "one_representation_per_sequence_id",
        "parquet_readback_exact",
    )
    validations = representation_manifest.get("validations", {})
    failed = [
        name
        for name in required_validations
        if validations.get(name) is not True
    ]

    if failed:
        raise ValueError(
            "Representation validations are incomplete: "
            + ", ".join(failed)
        )

    return representation_manifest, partition_manifest


def load_embeddings(
    config: dict[str, Any],
    representation_manifest: dict[str, Any],
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """Load and validate the frozen embedding matrix."""
    path = Path(config["input"]["embeddings"])
    dataset = config["dataset"]
    representation = config["representation"]
    sequence_id_column = dataset["sequence_id_column"]
    prefix = str(representation["embedding_column_prefix"])
    expected_dimension = int(representation["expected_dimension"])

    dataframe = pd.read_parquet(path)

    if sequence_id_column not in dataframe.columns:
        raise ValueError(
            f"Embedding table lacks '{sequence_id_column}'."
        )

    embedding_columns = sorted(
        column
        for column in dataframe.columns
        if str(column).startswith(prefix)
    )

    if len(embedding_columns) != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} embedding columns, "
            f"found {len(embedding_columns)}."
        )

    identifiers = (
        dataframe[sequence_id_column].astype(str).str.strip()
    )

    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError(
            "Embedding identifiers must be non-empty and unique."
        )

    matrix = dataframe[embedding_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Non-finite values were found in the embeddings."
        )

    expected_rows = int(representation_manifest["output"]["rows"])

    if len(dataframe) != expected_rows:
        raise ValueError(
            "Embedding row count differs from its manifest."
        )

    return (
        matrix,
        pd.Index(identifiers.to_numpy(), name=sequence_id_column),
        embedding_columns,
    )


def load_partition_index(config: dict[str, Any]) -> pd.DataFrame:
    """Load and validate the partition index."""
    path = Path(config["input"]["partition_index"])
    dataset = config["dataset"]
    dataframe = pd.read_csv(path, keep_default_na=False)

    required_columns = (
        "task_id",
        "regime_id",
        "seed",
        "split_id",
        "assignment_path",
        "assignment_sha256",
        "file_sha256",
        "validation_passed",
    )
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Partition index lacks columns: " + ", ".join(missing)
        )

    dataframe["task_id"] = (
        dataframe["task_id"].astype(str).str.strip()
    )
    dataframe["regime_id"] = (
        dataframe["regime_id"].astype(str).str.strip()
    )
    dataframe["seed"] = pd.to_numeric(
        dataframe["seed"], errors="raise"
    ).astype(int)
    dataframe["validation_passed"] = pd.to_numeric(
        dataframe["validation_passed"], errors="raise"
    ).astype(int)

    if (dataframe["validation_passed"] != 1).any():
        raise ValueError(
            "The partition index contains invalid partition sets."
        )

    tasks = list(dataset["included_tasks"])
    regimes = list(dataset["included_regimes"])
    dataframe = dataframe[
        dataframe["task_id"].isin(tasks)
        & dataframe["regime_id"].isin(regimes)
    ].copy()

    duplicate_mask = dataframe.duplicated(
        subset=["task_id", "regime_id", "seed"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicated task/regime/seed rows were found."
        )

    expected_seeds = int(
        dataset["expected_seeds_per_task_regime"]
    )
    counts = dataframe.groupby(
        ["task_id", "regime_id"], observed=True
    )["seed"].nunique()
    incomplete = counts[counts != expected_seeds]

    if not incomplete.empty:
        details = "; ".join(
            f"{task}/{regime}={count}"
            for (task, regime), count in incomplete.items()
        )
        raise ValueError(
            "Unexpected seed count per task/regime: " + details
        )

    return dataframe.sort_values(
        ["task_id", "regime_id", "seed"]
    ).reset_index(drop=True)


def parse_integer_filter(
    values: list[str] | None,
) -> set[int] | None:
    """Parse integers and inclusive ranges such as 0-9."""
    if not values:
        return None

    parsed: set[int] = set()

    for value in values:
        text = str(value).strip()

        if "-" in text:
            start_text, end_text = text.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise ValueError(f"Invalid seed range: {text}")

            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(text))

    return parsed


def filter_partition_index(
    dataframe: pd.DataFrame,
    *,
    tasks: list[str] | None,
    regimes: list[str] | None,
    seeds: set[int] | None,
) -> pd.DataFrame:
    """Apply optional command-line filters."""
    result = dataframe.copy()

    if tasks:
        missing = sorted(set(tasks) - set(result["task_id"]))

        if missing:
            raise ValueError(
                "Requested tasks were not found: " + ", ".join(missing)
            )

        result = result[result["task_id"].isin(tasks)]

    if regimes:
        missing = sorted(set(regimes) - set(result["regime_id"]))

        if missing:
            raise ValueError(
                "Requested regimes were not found: "
                + ", ".join(missing)
            )

        result = result[result["regime_id"].isin(regimes)]

    if seeds is not None:
        missing = sorted(seeds - set(result["seed"]))

        if missing:
            raise ValueError(
                "Requested seeds were not found: "
                + ", ".join(map(str, missing))
            )

        result = result[result["seed"].isin(seeds)]

    if result.empty:
        raise ValueError(
            "No partition sets remain after filtering."
        )

    return result.reset_index(drop=True)


def enabled_models(
    config: dict[str, Any],
    requested: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Return enabled model configurations."""
    available = {
        str(model_id): model_config
        for model_id, model_config in config["models"].items()
        if model_config.get("enabled", True)
    }

    if requested:
        missing = sorted(set(requested) - set(available))

        if missing:
            raise ValueError(
                "Requested models are missing or disabled: "
                + ", ".join(missing)
            )

        return {
            model_id: available[model_id]
            for model_id in requested
        }

    return available


def load_assignment(
    index_row: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate one split assignment."""
    path = Path(str(index_row["assignment_path"]))

    if not path.is_file():
        raise FileNotFoundError(f"Assignment file not found: {path}")

    if config["representation"].get(
        "verify_partition_file_sha256", True
    ):
        expected = str(index_row["file_sha256"])
        observed = sha256_file(path)

        if observed != expected:
            raise ValueError(
                f"Assignment SHA-256 mismatch: {path}"
            )

    dataframe = pd.read_csv(path, keep_default_na=False)
    dataset = config["dataset"]
    sequence_id_column = dataset["sequence_id_column"]
    label_column = dataset["label_column"]
    partition_column = dataset["partition_column"]

    required_columns = (
        "task_id",
        sequence_id_column,
        label_column,
        partition_column,
        "group_id",
        "seed",
        "split_id",
    )
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Assignment lacks columns: " + ", ".join(missing)
        )

    if set(dataframe["task_id"].astype(str)) != {
        str(index_row["task_id"])
    }:
        raise ValueError(
            "Assignment task differs from the partition index."
        )

    if set(
        pd.to_numeric(dataframe["seed"], errors="raise").astype(int)
    ) != {int(index_row["seed"])}:
        raise ValueError(
            "Assignment seed differs from the partition index."
        )

    if set(dataframe["split_id"].astype(str)) != {
        str(index_row["split_id"])
    }:
        raise ValueError(
            "Assignment split ID differs from the partition index."
        )

    dataframe[sequence_id_column] = (
        dataframe[sequence_id_column].astype(str).str.strip()
    )
    dataframe[label_column] = pd.to_numeric(
        dataframe[label_column], errors="raise"
    ).astype(np.int8)

    if (
        dataframe[sequence_id_column].eq("").any()
        or dataframe[sequence_id_column].duplicated().any()
    ):
        raise ValueError(
            "Assignment sequence IDs must be non-empty and unique."
        )

    if not set(dataframe[label_column].unique()).issubset({0, 1}):
        raise ValueError("Assignment labels must be binary.")

    if set(dataframe[partition_column].astype(str)) != set(
        dataset["expected_partitions"]
    ):
        raise ValueError(
            "Assignment partitions differ from the configuration."
        )

    for partition_name, subset in dataframe.groupby(
        partition_column, observed=True
    ):
        if subset.empty or subset[label_column].nunique() != 2:
            raise ValueError(
                f"Partition '{partition_name}' is empty or lacks a class."
            )

    group_partition_counts = dataframe.groupby(
        "group_id", observed=True
    )[partition_column].nunique()

    if (group_partition_counts > 1).any():
        raise ValueError(
            "At least one similarity group crosses partitions."
        )

    return dataframe


def map_partition(
    assignment: pd.DataFrame,
    partition_name: str,
    *,
    embedding_matrix: np.ndarray,
    embedding_index: pd.Index,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a partition to IDs, labels, groups, and embeddings."""
    dataset = config["dataset"]
    sequence_id_column = dataset["sequence_id_column"]
    label_column = dataset["label_column"]
    partition_column = dataset["partition_column"]

    subset = (
        assignment[
            assignment[partition_column] == partition_name
        ]
        .sort_values(sequence_id_column)
        .reset_index(drop=True)
    )

    identifiers = subset[sequence_id_column].astype(str).to_numpy()
    labels = subset[label_column].to_numpy(dtype=np.int8)
    groups = subset["group_id"].astype(str).to_numpy()
    positions = embedding_index.get_indexer(identifiers)

    if (positions < 0).any():
        missing_ids = identifiers[positions < 0][:10]
        raise ValueError(
            "Partition IDs are missing from embeddings: "
            + ", ".join(map(str, missing_ids))
        )

    return identifiers, labels, groups, embedding_matrix[positions]


def model_variant_parameters(
    model_config: dict[str, Any],
    variant: dict[str, Any],
    model_seed: int,
) -> dict[str, Any]:
    """Merge fixed and variant-specific parameters."""
    parameters = {
        **model_config.get("fixed_parameters", {}),
        **variant.get("parameters", {}),
    }

    if model_config["estimator"] in {
        "logistic_regression",
        "random_forest",
        "linear_svm",
    }:
        parameters["random_state"] = model_seed

    return parameters


def build_estimator(
    model_config: dict[str, Any],
    parameters: dict[str, Any],
) -> BaseEstimator:
    """Build a leakage-safe estimator."""
    estimator_type = model_config["estimator"]

    if estimator_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(**parameters),
                ),
            ]
        )

    if estimator_type == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(**parameters)),
            ]
        )

    if estimator_type == "random_forest":
        return RandomForestClassifier(**parameters)

    raise ValueError(
        f"Generic estimator builder does not support '{estimator_type}'."
    )


def positive_scores(
    estimator: BaseEstimator,
    matrix: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Extract positive-class probability or decision scores."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(matrix)
        classes = np.asarray(estimator.classes_)
        positions = np.where(classes == 1)[0]

        if positions.size != 1:
            raise ValueError(
                "Could not identify the positive probability column."
            )

        scores = probabilities[:, int(positions[0])].astype(
            np.float64, copy=False
        )
        score_type = "probability"

    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(
            estimator.decision_function(matrix),
            dtype=np.float64,
        ).reshape(-1)
        score_type = "decision_function"

    else:
        raise ValueError(
            "Estimator exposes neither predict_proba nor decision_function."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores were produced.")

    return scores, score_type


def fixed_threshold(score_type: str, config: dict[str, Any]) -> float:
    """Return the predeclared threshold without optimization."""
    thresholds = config["evaluation"]["fixed_thresholds"]

    if score_type not in thresholds:
        raise ValueError(
            f"No fixed threshold configured for '{score_type}'."
        )

    return float(thresholds[score_type])


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    score_type: str,
    threshold: float,
) -> dict[str, Any]:
    """Calculate threshold-free and fixed-threshold metrics."""
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else None
    )

    return {
        "n": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "prevalence": float(labels.mean()),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "threshold": threshold,
        "prediction_positive_fraction": float(predictions.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": specificity,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "brier_score": (
            float(brier_score_loss(labels, scores))
            if score_type == "probability"
            else None
        ),
    }


def knn_probabilities(
    neighbor_labels: np.ndarray,
    distances: np.ndarray,
    *,
    n_neighbors: int,
    weights: str,
) -> np.ndarray:
    """Calculate binary KNN probabilities from a shared neighbor query."""
    labels = neighbor_labels[:, :n_neighbors].astype(np.float64)
    local_distances = distances[:, :n_neighbors].astype(np.float64)

    if weights == "uniform":
        return labels.mean(axis=1)

    if weights != "distance":
        raise ValueError(f"Unsupported KNN weights: {weights}")

    zero_mask = local_distances == 0.0
    rows_with_zero = zero_mask.any(axis=1)
    probabilities = np.empty(labels.shape[0], dtype=np.float64)

    if rows_with_zero.any():
        zero_weights = zero_mask[rows_with_zero].astype(np.float64)
        probabilities[rows_with_zero] = (
            (labels[rows_with_zero] * zero_weights).sum(axis=1)
            / zero_weights.sum(axis=1)
        )

    rows_without_zero = ~rows_with_zero

    if rows_without_zero.any():
        inverse = 1.0 / local_distances[rows_without_zero]
        probabilities[rows_without_zero] = (
            (labels[rows_without_zero] * inverse).sum(axis=1)
            / inverse.sum(axis=1)
        )

    return probabilities


def verify_scaler_train_only(
    estimator: BaseEstimator,
    train_size: int,
) -> int | None:
    """Verify that a selected scaler saw exactly the training samples."""
    if not isinstance(estimator, Pipeline):
        return None

    if "scaler" not in estimator.named_steps:
        return None

    observed = estimator.named_steps["scaler"].n_samples_seen_
    count = int(np.asarray(observed).max())

    if count != train_size:
        raise RuntimeError(
            "StandardScaler was not fitted on exactly the training rows."
        )

    return count


def evaluate_generic_variant(
    *,
    model_config: dict[str, Any],
    variant: dict[str, Any],
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    evaluation_matrix: np.ndarray,
    evaluation_labels: np.ndarray,
    model_seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit and evaluate one fixed non-KNN variant."""
    parameters = model_variant_parameters(
        model_config,
        variant,
        model_seed,
    )
    estimator = build_estimator(model_config, parameters)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.monotonic()
        estimator.fit(train_matrix, train_labels)
        fit_seconds = time.monotonic() - started

        score_started = time.monotonic()
        scores, score_type = positive_scores(
            estimator,
            evaluation_matrix,
        )
        score_seconds = time.monotonic() - score_started

    warning_messages = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
    ]
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )

    if (
        convergence_warning
        and config["execution"].get(
            "reject_convergence_warnings", True
        )
    ):
        raise RuntimeError(
            "Variant emitted a ConvergenceWarning: "
            + " | ".join(warning_messages)
        )

    scaler_count = verify_scaler_train_only(
        estimator,
        len(train_labels),
    )
    threshold = fixed_threshold(score_type, config)
    metrics = calculate_metrics(
        evaluation_labels,
        scores,
        score_type=score_type,
        threshold=threshold,
    )

    return {
        "parameters": parameters,
        "estimator": estimator,
        "scores": scores,
        "score_type": score_type,
        "threshold": threshold,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "warning_messages": warning_messages,
        "scaler_n_samples_seen": scaler_count,
    }


def grid_rows_for_knn(
    *,
    task_id: str,
    reference_regime: str,
    seed: int,
    split_id: str,
    model_id: str,
    model_config: dict[str, Any],
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all KNN variants using one exact neighbor query."""
    variants = model_config["variants"]
    maximum_neighbors = max(
        int(variant["parameters"]["n_neighbors"])
        for variant in variants
    )

    if maximum_neighbors > len(train_labels):
        raise ValueError(
            "A KNN variant requests more neighbors than training rows."
        )

    scaler = StandardScaler()
    fit_started = time.monotonic()
    scaled_train = scaler.fit_transform(train_matrix)
    scaled_validation = scaler.transform(validation_matrix)

    fixed = model_config.get("fixed_parameters", {})
    neighbors = NearestNeighbors(
        n_neighbors=maximum_neighbors,
        algorithm=str(fixed.get("algorithm", "brute")),
        metric=str(fixed.get("metric", "euclidean")),
        n_jobs=int(fixed.get("n_jobs", -1)),
    )
    neighbors.fit(scaled_train)
    shared_fit_seconds = time.monotonic() - fit_started
    verify_scaler_count = int(np.asarray(scaler.n_samples_seen_).max())

    if verify_scaler_count != len(train_labels):
        raise RuntimeError(
            "KNN StandardScaler was not fitted on train only."
        )

    query_started = time.monotonic()
    distances, indices = neighbors.kneighbors(
        scaled_validation,
        n_neighbors=maximum_neighbors,
        return_distance=True,
    )
    neighbor_labels = train_labels[indices]
    shared_query_seconds = time.monotonic() - query_started
    rows: list[dict[str, Any]] = []

    for variant_index, variant in enumerate(variants):
        variant_id = str(variant["id"])
        parameters = {
            **fixed,
            **variant.get("parameters", {}),
        }

        try:
            score_started = time.monotonic()
            scores = knn_probabilities(
                neighbor_labels,
                distances,
                n_neighbors=int(parameters["n_neighbors"]),
                weights=str(parameters["weights"]),
            )
            score_seconds = time.monotonic() - score_started
            score_type = "probability"
            threshold = fixed_threshold(score_type, config)
            metrics = calculate_metrics(
                validation_labels,
                scores,
                score_type=score_type,
                threshold=threshold,
            )

            rows.append(
                grid_result_row(
                    task_id=task_id,
                    reference_regime=reference_regime,
                    seed=seed,
                    split_id=split_id,
                    model_id=model_id,
                    variant_id=variant_id,
                    parameters=parameters,
                    status="valid",
                    score_type=score_type,
                    metrics=metrics,
                    fit_seconds=(
                        shared_fit_seconds
                        if variant_index == 0
                        else 0.0
                    ),
                    score_seconds=(
                        shared_query_seconds
                        if variant_index == 0
                        else 0.0
                    )
                    + score_seconds,
                )
            )

        except Exception as error:
            rows.append(
                grid_result_row(
                    task_id=task_id,
                    reference_regime=reference_regime,
                    seed=seed,
                    split_id=split_id,
                    model_id=model_id,
                    variant_id=variant_id,
                    parameters=parameters,
                    status="failed",
                    score_type="probability",
                    metrics=None,
                    error=error,
                )
            )

    return rows


def grid_result_row(
    *,
    task_id: str,
    reference_regime: str,
    seed: int,
    split_id: str,
    model_id: str,
    variant_id: str,
    parameters: dict[str, Any],
    status: str,
    score_type: str,
    metrics: dict[str, Any] | None,
    fit_seconds: float | None = None,
    score_seconds: float | None = None,
    warning_messages: list[str] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Build one fixed-grid validation record."""
    warning_messages = warning_messages or []
    metrics = metrics or {}

    return {
        "task_id": task_id,
        "reference_regime": reference_regime,
        "seed": seed,
        "split_id": split_id,
        "model_id": model_id,
        "variant_id": variant_id,
        "parameters_json": canonical_json(parameters),
        "status": status,
        "score_type": score_type,
        "validation_n": metrics.get("n"),
        "validation_positive_count": metrics.get("positive_count"),
        "validation_negative_count": metrics.get("negative_count"),
        "validation_prevalence": metrics.get("prevalence"),
        "validation_average_precision": metrics.get(
            "average_precision"
        ),
        "validation_roc_auc": metrics.get("roc_auc"),
        "validation_threshold": metrics.get("threshold"),
        "validation_accuracy": metrics.get("accuracy"),
        "validation_balanced_accuracy": metrics.get(
            "balanced_accuracy"
        ),
        "validation_mcc": metrics.get("mcc"),
        "validation_f1": metrics.get("f1"),
        "validation_precision": metrics.get("precision"),
        "validation_recall_sensitivity": metrics.get(
            "recall_sensitivity"
        ),
        "validation_specificity": metrics.get("specificity"),
        "validation_brier_score": metrics.get("brier_score"),
        "fit_seconds": fit_seconds,
        "validation_score_seconds": score_seconds,
        "warning_count": len(warning_messages),
        "warnings": " | ".join(warning_messages),
        "error": (
            f"{type(error).__name__}: {error}"
            if error is not None
            else ""
        ),
    }


def run_grid_model(
    *,
    index_row: pd.Series,
    model_id: str,
    model_config: dict[str, Any],
    config: dict[str, Any],
    configuration_sha256: str,
    embedding_matrix: np.ndarray,
    embedding_index: pd.Index,
    representation_manifest: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate all fixed variants on validation for one reference split."""
    task_id = str(index_row["task_id"])
    regime_id = str(index_row["regime_id"])
    seed = int(index_row["seed"])
    split_id = str(index_row["split_id"])
    reference_regime = str(config["selection"]["reference_regime"])

    if regime_id != reference_regime:
        raise ValueError(
            "Grid evaluation is restricted to the reference regime."
        )

    output_root = Path(config["output"]["run_root"])
    run_directory = (
        output_root
        / "grid"
        / task_id
        / regime_id
        / f"seed_{seed:03d}"
        / model_id
    )
    results_path = run_directory / "grid_validation.csv"
    manifest_path = run_directory / "grid_manifest.json"
    failure_path = run_directory / "failure.json"

    provenance = {
        "configuration_sha256": configuration_sha256,
        "embedding_sha256": representation_manifest["output"]["sha256"],
        "assignment_file_sha256": str(index_row["file_sha256"]),
    }

    if run_directory.exists():
        if overwrite:
            shutil.rmtree(run_directory)
        elif (
            config["execution"].get("resume_existing_runs", True)
            and manifest_path.is_file()
            and results_path.is_file()
        ):
            manifest = load_json(manifest_path)

            if (
                manifest.get("status") == "complete"
                and manifest.get("provenance") == provenance
            ):
                return {
                    "status": "skipped_existing",
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                }

            raise ValueError(
                f"Existing grid run is incompatible: {run_directory}"
            )
        else:
            raise FileExistsError(
                f"Grid run directory already exists: {run_directory}"
            )

    run_directory.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started = time.monotonic()

    try:
        assignment = load_assignment(index_row, config)
        (
            train_ids,
            train_labels,
            train_groups,
            train_matrix,
        ) = map_partition(
            assignment,
            "train",
            embedding_matrix=embedding_matrix,
            embedding_index=embedding_index,
            config=config,
        )
        (
            validation_ids,
            validation_labels,
            validation_groups,
            validation_matrix,
        ) = map_partition(
            assignment,
            "validation",
            embedding_matrix=embedding_matrix,
            embedding_index=embedding_index,
            config=config,
        )

        del train_ids, train_groups, validation_ids, validation_groups

        if model_config["estimator"] == "knn":
            rows = grid_rows_for_knn(
                task_id=task_id,
                reference_regime=reference_regime,
                seed=seed,
                split_id=split_id,
                model_id=model_id,
                model_config=model_config,
                train_matrix=train_matrix,
                train_labels=train_labels,
                validation_matrix=validation_matrix,
                validation_labels=validation_labels,
                config=config,
            )
        else:
            rows = []
            model_seed = int(config["training"]["model_seed"])

            for variant in model_config["variants"]:
                variant_id = str(variant["id"])
                parameters = model_variant_parameters(
                    model_config,
                    variant,
                    model_seed,
                )

                try:
                    result = evaluate_generic_variant(
                        model_config=model_config,
                        variant=variant,
                        train_matrix=train_matrix,
                        train_labels=train_labels,
                        evaluation_matrix=validation_matrix,
                        evaluation_labels=validation_labels,
                        model_seed=model_seed,
                        config=config,
                    )
                    rows.append(
                        grid_result_row(
                            task_id=task_id,
                            reference_regime=reference_regime,
                            seed=seed,
                            split_id=split_id,
                            model_id=model_id,
                            variant_id=variant_id,
                            parameters=result["parameters"],
                            status="valid",
                            score_type=result["score_type"],
                            metrics=result["metrics"],
                            fit_seconds=result["fit_seconds"],
                            score_seconds=result["score_seconds"],
                            warning_messages=result[
                                "warning_messages"
                            ],
                        )
                    )

                except Exception as error:
                    rows.append(
                        grid_result_row(
                            task_id=task_id,
                            reference_regime=reference_regime,
                            seed=seed,
                            split_id=split_id,
                            model_id=model_id,
                            variant_id=variant_id,
                            parameters=parameters,
                            status="failed",
                            score_type=str(
                                model_config.get("score_type", "")
                            ),
                            metrics=None,
                            error=error,
                        )
                    )

        results = pd.DataFrame(rows, columns=GRID_RESULT_COLUMNS)
        results.to_csv(results_path, index=False)

        valid_count = int((results["status"] == "valid").sum())
        expected_count = len(model_config["variants"])

        if valid_count != expected_count:
            raise RuntimeError(
                f"Only {valid_count}/{expected_count} fixed variants "
                "completed successfully."
            )

        manifest = {
            "schema_version": "2.0",
            "status": "complete",
            "stage": "fixed_grid_validation",
            "created_at_utc": utc_now(),
            "started_at_utc": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "task_id": task_id,
            "reference_regime": reference_regime,
            "seed": seed,
            "split_id": split_id,
            "model_id": model_id,
            "variant_count": expected_count,
            "valid_variant_count": valid_count,
            "test_loaded": False,
            "test_evaluations": 0,
            "model_seed": int(config["training"]["model_seed"]),
            "provenance": provenance,
            "output": {
                "path": str(results_path),
                "sha256": sha256_file(results_path),
                "rows": int(len(results)),
            },
        }
        write_json(manifest, manifest_path)

        if failure_path.exists():
            failure_path.unlink()

        return {
            "status": "completed",
            "manifest": manifest,
            "manifest_path": str(manifest_path),
        }

    except Exception as error:
        failure = {
            "schema_version": "2.0",
            "status": "failed",
            "stage": "fixed_grid_validation",
            "created_at_utc": utc_now(),
            "task_id": task_id,
            "regime_id": regime_id,
            "seed": seed,
            "split_id": split_id,
            "model_id": model_id,
            "variant_id": None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "provenance": provenance,
        }
        write_json(failure, failure_path)
        raise

    finally:
        gc.collect()


def collect_grid_results(
    run_root: Path,
    configuration_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect grid records and failures for the active configuration."""
    frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []

    for manifest_path in run_root.rglob("grid_manifest.json"):
        manifest = load_json(manifest_path)

        if manifest.get("provenance", {}).get(
            "configuration_sha256"
        ) != configuration_sha256:
            continue

        output_path = Path(manifest["output"]["path"])

        if output_path.is_file():
            frames.append(pd.read_csv(output_path, keep_default_na=False))

    for failure_path in run_root.rglob("failure.json"):
        failure = load_json(failure_path)

        if failure.get("stage") != "fixed_grid_validation":
            continue

        if failure.get("provenance", {}).get(
            "configuration_sha256"
        ) != configuration_sha256:
            continue

        failure_rows.append(
            {
                "stage": failure.get("stage"),
                "task_id": failure.get("task_id"),
                "regime_id": failure.get("regime_id"),
                "seed": failure.get("seed"),
                "split_id": failure.get("split_id"),
                "model_id": failure.get("model_id"),
                "variant_id": failure.get("variant_id"),
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
                "failure_path": str(failure_path),
            }
        )

    results = (
        pd.concat(frames, ignore_index=True)
        if frames
        else empty_dataframe(GRID_RESULT_COLUMNS)
    )
    failures = (
        pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)
        if failure_rows
        else empty_dataframe(FAILURE_COLUMNS)
    )

    if not results.empty:
        results = results.sort_values(
            ["task_id", "model_id", "variant_id", "seed"]
        ).reset_index(drop=True)

    return results, failures


def select_representative_variants(
    *,
    grid_results: pd.DataFrame,
    config: dict[str, Any],
    allow_partial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one variant per task/model using aggregate validation AP."""
    valid = grid_results[
        grid_results["status"] == "valid"
    ].copy()

    if valid.empty:
        raise ValueError(
            "No valid fixed-grid validation records are available."
        )

    valid["validation_average_precision"] = pd.to_numeric(
        valid["validation_average_precision"],
        errors="raise",
    )
    valid["seed"] = pd.to_numeric(valid["seed"], errors="raise").astype(
        int
    )
    expected_seeds = int(
        config["dataset"]["expected_seeds_per_task_regime"]
    )
    require_complete = bool(
        config["selection"].get(
            "require_complete_reference_grid", True
        )
    )

    grouped = valid.groupby(
        [
            "task_id",
            "model_id",
            "variant_id",
            "parameters_json",
            "reference_regime",
        ],
        observed=True,
        sort=True,
    )
    summary_rows: list[dict[str, Any]] = []

    for keys, group in grouped:
        (
            task_id,
            model_id,
            variant_id,
            parameters_json,
            reference_regime,
        ) = keys
        values = group["validation_average_precision"]
        seed_count = int(group["seed"].nunique())

        if (
            require_complete
            and not allow_partial
            and seed_count != expected_seeds
        ):
            raise ValueError(
                "Incomplete reference grid for "
                f"{task_id}/{model_id}/{variant_id}: "
                f"{seed_count}/{expected_seeds} seeds."
            )

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        summary_rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "parameters_json": parameters_json,
                "reference_regime": reference_regime,
                "selection_metric": "validation_average_precision",
                "selection_seed_count": seed_count,
                "validation_ap_median": float(values.median()),
                "validation_ap_q1": q1,
                "validation_ap_q3": q3,
                "validation_ap_iqr": q3 - q1,
                "validation_ap_mean": float(values.mean()),
                "validation_ap_standard_deviation": float(
                    values.std(ddof=1)
                    if len(values) > 1
                    else 0.0
                ),
                "validation_ap_minimum": float(values.min()),
                "validation_ap_maximum": float(values.max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    ranked_frames: list[pd.DataFrame] = []

    for (_, _), group in summary.groupby(
        ["task_id", "model_id"],
        observed=True,
        sort=True,
    ):
        ranked = group.sort_values(
            [
                "validation_ap_median",
                "validation_ap_mean",
                "validation_ap_iqr",
                "variant_id",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
        ranked["selected"] = (ranked["selection_rank"] == 1).astype(int)
        ranked_frames.append(ranked)

    ranked_summary = pd.concat(ranked_frames, ignore_index=True)
    ranked_summary = ranked_summary[
        SELECTED_VARIANT_COLUMNS
    ].sort_values(
        ["task_id", "model_id", "selection_rank"]
    ).reset_index(drop=True)
    selected = ranked_summary[
        ranked_summary["selected"] == 1
    ].copy()

    expected_models = {
        model_id
        for model_id, model_config in config["models"].items()
        if model_config.get("enabled", True)
    }
    expected_tasks = set(config["dataset"]["included_tasks"])
    expected_pairs = {
        (task_id, model_id)
        for task_id in expected_tasks
        for model_id in expected_models
    }
    observed_pairs = set(
        zip(selected["task_id"], selected["model_id"])
    )

    if not allow_partial and observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise ValueError(
            "Representative variants are missing for: "
            + "; ".join(
                f"{task}/{model}" for task, model in missing
            )
        )

    return ranked_summary, selected


def find_variant(
    model_config: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    """Find a configured variant by ID."""
    matches = [
        variant
        for variant in model_config["variants"]
        if str(variant["id"]) == variant_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve configured variant '{variant_id}'."
        )

    return matches[0]


def fit_selected_knn(
    *,
    model_config: dict[str, Any],
    variant: dict[str, Any],
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> dict[str, Any]:
    """Fit one frozen KNN variant and score validation and test."""
    parameters = {
        **model_config.get("fixed_parameters", {}),
        **variant.get("parameters", {}),
    }
    n_neighbors = int(parameters["n_neighbors"])

    if n_neighbors > len(train_labels):
        raise ValueError(
            "Selected KNN variant requests too many neighbors."
        )

    scaler = StandardScaler()
    started = time.monotonic()
    scaled_train = scaler.fit_transform(train_matrix)
    scaled_validation = scaler.transform(validation_matrix)
    scaled_test = scaler.transform(test_matrix)
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=str(parameters.get("algorithm", "brute")),
        metric=str(parameters.get("metric", "euclidean")),
        n_jobs=int(parameters.get("n_jobs", -1)),
    )
    neighbors.fit(scaled_train)
    fit_seconds = time.monotonic() - started

    scaler_count = int(np.asarray(scaler.n_samples_seen_).max())

    if scaler_count != len(train_labels):
        raise RuntimeError(
            "KNN scaler was not fitted on train only."
        )

    validation_started = time.monotonic()
    validation_distances, validation_indices = neighbors.kneighbors(
        scaled_validation,
        n_neighbors=n_neighbors,
        return_distance=True,
    )
    validation_scores = knn_probabilities(
        train_labels[validation_indices],
        validation_distances,
        n_neighbors=n_neighbors,
        weights=str(parameters["weights"]),
    )
    validation_score_seconds = time.monotonic() - validation_started

    test_started = time.monotonic()
    test_distances, test_indices = neighbors.kneighbors(
        scaled_test,
        n_neighbors=n_neighbors,
        return_distance=True,
    )
    test_scores = knn_probabilities(
        train_labels[test_indices],
        test_distances,
        n_neighbors=n_neighbors,
        weights=str(parameters["weights"]),
    )
    test_score_seconds = time.monotonic() - test_started

    return {
        "parameters": parameters,
        "score_type": "probability",
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "fit_seconds": fit_seconds,
        "validation_score_seconds": validation_score_seconds,
        "test_score_seconds": test_score_seconds,
        "scaler_n_samples_seen": scaler_count,
        "warning_messages": [],
    }


def fit_selected_generic(
    *,
    model_config: dict[str, Any],
    variant: dict[str, Any],
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    test_matrix: np.ndarray,
    model_seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit one frozen non-KNN variant and score validation and test."""
    parameters = model_variant_parameters(
        model_config,
        variant,
        model_seed,
    )
    estimator = build_estimator(model_config, parameters)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.monotonic()
        estimator.fit(train_matrix, train_labels)
        fit_seconds = time.monotonic() - started

        validation_started = time.monotonic()
        validation_scores, validation_score_type = positive_scores(
            estimator,
            validation_matrix,
        )
        validation_score_seconds = (
            time.monotonic() - validation_started
        )

        # Test is scored only after the frozen estimator has been fitted.
        test_started = time.monotonic()
        test_scores, test_score_type = positive_scores(
            estimator,
            test_matrix,
        )
        test_score_seconds = time.monotonic() - test_started

    if validation_score_type != test_score_type:
        raise RuntimeError(
            "Score type changed between validation and test."
        )

    warning_messages = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
    ]
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )

    if (
        convergence_warning
        and config["execution"].get(
            "reject_convergence_warnings", True
        )
    ):
        raise RuntimeError(
            "Selected variant emitted a ConvergenceWarning: "
            + " | ".join(warning_messages)
        )

    scaler_count = verify_scaler_train_only(
        estimator,
        len(train_labels),
    )

    return {
        "parameters": parameters,
        "score_type": validation_score_type,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "fit_seconds": fit_seconds,
        "validation_score_seconds": validation_score_seconds,
        "test_score_seconds": test_score_seconds,
        "scaler_n_samples_seen": scaler_count,
        "warning_messages": warning_messages,
    }


def prediction_table(
    *,
    task_id: str,
    regime_id: str,
    seed: int,
    split_id: str,
    model_id: str,
    variant_id: str,
    sequence_ids: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    score_type: str,
    threshold: float,
) -> pd.DataFrame:
    """Build one fixed-variant prediction table."""
    return pd.DataFrame(
        {
            "task_id": task_id,
            "regime_id": regime_id,
            "seed": seed,
            "split_id": split_id,
            "model_id": model_id,
            "variant_id": variant_id,
            "partition": "test",
            "sequence_id": sequence_ids,
            "group_id": group_ids,
            "amp_label": labels,
            "score": scores,
            "score_type": score_type,
            "threshold": threshold,
            "predicted_label": (scores >= threshold).astype(np.int8),
        }
    )


def run_final_model(
    *,
    index_row: pd.Series,
    selected_row: pd.Series,
    model_id: str,
    model_config: dict[str, Any],
    config: dict[str, Any],
    configuration_sha256: str,
    selection_file_sha256: str,
    embedding_matrix: np.ndarray,
    embedding_index: pd.Index,
    representation_manifest: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate one globally selected variant on one held-out split."""
    task_id = str(index_row["task_id"])
    regime_id = str(index_row["regime_id"])
    seed = int(index_row["seed"])
    split_id = str(index_row["split_id"])
    variant_id = str(selected_row["variant_id"])
    variant = find_variant(model_config, variant_id)
    output_root = Path(config["output"]["run_root"])
    run_directory = (
        output_root
        / "final"
        / task_id
        / regime_id
        / f"seed_{seed:03d}"
        / model_id
    )
    manifest_path = run_directory / "run_manifest.json"
    predictions_path = run_directory / "test_predictions.parquet"
    failure_path = run_directory / "failure.json"
    provenance = {
        "configuration_sha256": configuration_sha256,
        "selection_file_sha256": selection_file_sha256,
        "embedding_sha256": representation_manifest["output"]["sha256"],
        "assignment_file_sha256": str(index_row["file_sha256"]),
    }

    if run_directory.exists():
        if overwrite:
            shutil.rmtree(run_directory)
        elif (
            config["execution"].get("resume_existing_runs", True)
            and manifest_path.is_file()
        ):
            manifest = load_json(manifest_path)

            if (
                manifest.get("status") == "complete"
                and manifest.get("provenance") == provenance
                and manifest.get("variant_id") == variant_id
            ):
                return {
                    "status": "skipped_existing",
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                }

            raise ValueError(
                f"Existing final run is incompatible: {run_directory}"
            )
        else:
            raise FileExistsError(
                f"Final run directory already exists: {run_directory}"
            )

    run_directory.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started = time.monotonic()

    try:
        assignment = load_assignment(index_row, config)
        train_ids, train_labels, train_groups, train_matrix = map_partition(
            assignment,
            "train",
            embedding_matrix=embedding_matrix,
            embedding_index=embedding_index,
            config=config,
        )
        (
            validation_ids,
            validation_labels,
            validation_groups,
            validation_matrix,
        ) = map_partition(
            assignment,
            "validation",
            embedding_matrix=embedding_matrix,
            embedding_index=embedding_index,
            config=config,
        )
        test_ids, test_labels, test_groups, test_matrix = map_partition(
            assignment,
            "test",
            embedding_matrix=embedding_matrix,
            embedding_index=embedding_index,
            config=config,
        )
        del train_ids, train_groups, validation_ids, validation_groups

        model_seed = int(config["training"]["model_seed"])

        if model_config["estimator"] == "knn":
            result = fit_selected_knn(
                model_config=model_config,
                variant=variant,
                train_matrix=train_matrix,
                train_labels=train_labels,
                validation_matrix=validation_matrix,
                test_matrix=test_matrix,
            )
        else:
            result = fit_selected_generic(
                model_config=model_config,
                variant=variant,
                train_matrix=train_matrix,
                train_labels=train_labels,
                validation_matrix=validation_matrix,
                test_matrix=test_matrix,
                model_seed=model_seed,
                config=config,
            )

        score_type = str(result["score_type"])
        threshold = fixed_threshold(score_type, config)
        validation_metrics = calculate_metrics(
            validation_labels,
            result["validation_scores"],
            score_type=score_type,
            threshold=threshold,
        )
        test_metrics = calculate_metrics(
            test_labels,
            result["test_scores"],
            score_type=score_type,
            threshold=threshold,
        )

        outputs: dict[str, Any] = {}

        if config["execution"].get("write_test_predictions", True):
            predictions = prediction_table(
                task_id=task_id,
                regime_id=regime_id,
                seed=seed,
                split_id=split_id,
                model_id=model_id,
                variant_id=variant_id,
                sequence_ids=test_ids,
                group_ids=test_groups,
                labels=test_labels,
                scores=result["test_scores"],
                score_type=score_type,
                threshold=threshold,
            )
            predictions.to_parquet(
                predictions_path,
                index=False,
                compression=str(
                    config["execution"].get(
                        "parquet_compression", "zstd"
                    )
                ),
            )
            outputs["test_predictions"] = {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
                "rows": int(len(predictions)),
                "size_bytes": int(predictions_path.stat().st_size),
            }

        manifest = {
            "schema_version": "2.0",
            "status": "complete",
            "stage": "final_fixed_variant_evaluation",
            "created_at_utc": utc_now(),
            "started_at_utc": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "task_id": task_id,
            "regime_id": regime_id,
            "seed": seed,
            "split_id": split_id,
            "model_id": model_id,
            "variant_id": variant_id,
            "parameters": result["parameters"],
            "estimator": model_config["estimator"],
            "preprocessing": model_config["preprocessing"],
            "score_type": score_type,
            "calibration": model_config.get(
                "calibration", "not_applicable"
            ),
            "fixed_threshold": threshold,
            "selection": {
                "scope": "one variant per task and model family",
                "reference_regime": config["selection"][
                    "reference_regime"
                ],
                "metric": "median validation average precision",
                "selected_before_final_test_evaluation": True,
                "per_split_hyperparameter_tuning": False,
                "per_split_threshold_tuning": False,
            },
            "data_counts": {
                "train": int(len(train_labels)),
                "validation": int(len(validation_labels)),
                "test": int(len(test_labels)),
            },
            "leakage_controls": {
                "scaler_fitted_on_train_only": (
                    model_config["preprocessing"]
                    == "standard_scaler"
                ),
                "scaler_n_samples_seen": result[
                    "scaler_n_samples_seen"
                ],
                "test_used_for_variant_selection": False,
                "test_used_for_threshold_selection": False,
                "test_scored_once": True,
                "refit_on_train_validation": False,
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "timing": {
                "fit_seconds": result["fit_seconds"],
                "validation_score_seconds": result[
                    "validation_score_seconds"
                ],
                "test_score_seconds": result["test_score_seconds"],
            },
            "warnings": result["warning_messages"],
            "provenance": provenance,
            "outputs": outputs,
        }
        write_json(manifest, manifest_path)

        if failure_path.exists():
            failure_path.unlink()

        return {
            "status": "completed",
            "manifest": manifest,
            "manifest_path": str(manifest_path),
        }

    except Exception as error:
        failure = {
            "schema_version": "2.0",
            "status": "failed",
            "stage": "final_fixed_variant_evaluation",
            "created_at_utc": utc_now(),
            "task_id": task_id,
            "regime_id": regime_id,
            "seed": seed,
            "split_id": split_id,
            "model_id": model_id,
            "variant_id": variant_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "provenance": provenance,
        }
        write_json(failure, failure_path)
        raise

    finally:
        gc.collect()


def performance_row(
    manifest: dict[str, Any],
    manifest_path: Path,
    partition: str,
) -> dict[str, Any]:
    """Flatten one final performance record."""
    metrics = manifest[f"{partition}_metrics"]

    return {
        "task_id": manifest["task_id"],
        "regime_id": manifest["regime_id"],
        "seed": manifest["seed"],
        "split_id": manifest["split_id"],
        "model_id": manifest["model_id"],
        "variant_id": manifest["variant_id"],
        "parameters_json": canonical_json(manifest["parameters"]),
        "partition": partition,
        "score_type": manifest["score_type"],
        "threshold": manifest["fixed_threshold"],
        **metrics,
        "run_manifest_path": str(manifest_path),
    }


def collect_final_results(
    run_root: Path,
    configuration_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect final performance and failures."""
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for manifest_path in run_root.rglob("run_manifest.json"):
        manifest = load_json(manifest_path)

        if manifest.get("stage") != "final_fixed_variant_evaluation":
            continue

        if manifest.get("provenance", {}).get(
            "configuration_sha256"
        ) != configuration_sha256:
            continue

        if manifest.get("status") != "complete":
            continue

        rows.extend(
            [
                performance_row(manifest, manifest_path, "validation"),
                performance_row(manifest, manifest_path, "test"),
            ]
        )

    for failure_path in run_root.rglob("failure.json"):
        failure = load_json(failure_path)

        if failure.get("stage") != "final_fixed_variant_evaluation":
            continue

        if failure.get("provenance", {}).get(
            "configuration_sha256"
        ) != configuration_sha256:
            continue

        failure_rows.append(
            {
                "stage": failure.get("stage"),
                "task_id": failure.get("task_id"),
                "regime_id": failure.get("regime_id"),
                "seed": failure.get("seed"),
                "split_id": failure.get("split_id"),
                "model_id": failure.get("model_id"),
                "variant_id": failure.get("variant_id"),
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
                "failure_path": str(failure_path),
            }
        )

    performance = (
        pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)
        if rows
        else empty_dataframe(PERFORMANCE_COLUMNS)
    )
    failures = (
        pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)
        if failure_rows
        else empty_dataframe(FAILURE_COLUMNS)
    )

    if not performance.empty:
        performance = performance.sort_values(
            [
                "task_id",
                "regime_id",
                "seed",
                "model_id",
                "partition",
            ]
        ).reset_index(drop=True)

    return performance, failures


def write_summary_tables(
    *,
    config: dict[str, Any],
    configuration_sha256: str,
) -> dict[str, Any]:
    """Write global grid, selection, performance, and failure summaries."""
    output = config["output"]
    run_root = Path(output["run_root"])
    grid_results, grid_failures = collect_grid_results(
        run_root,
        configuration_sha256,
    )
    final_performance, final_failures = collect_final_results(
        run_root,
        configuration_sha256,
    )
    failures = pd.concat(
        [grid_failures, final_failures],
        ignore_index=True,
    )

    if failures.empty:
        failures = empty_dataframe(FAILURE_COLUMNS)

    paths = {
        "grid_validation": Path(output["grid_validation_summary"]),
        "grid_ranking": Path(output["grid_ranking_summary"]),
        "selected_variants": Path(output["selected_variants"]),
        "performance": Path(output["performance_summary"]),
        "failures": Path(output["failure_summary"]),
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    grid_results.to_csv(paths["grid_validation"], index=False)
    final_performance.to_csv(paths["performance"], index=False)
    failures.to_csv(paths["failures"], index=False)

    if Path(output["grid_ranking_summary"]).is_file():
        ranking = pd.read_csv(
            output["grid_ranking_summary"],
            keep_default_na=False,
        )
    else:
        ranking = empty_dataframe(SELECTED_VARIANT_COLUMNS)
        ranking.to_csv(paths["grid_ranking"], index=False)

    if Path(output["selected_variants"]).is_file():
        selected = pd.read_csv(
            output["selected_variants"],
            keep_default_na=False,
        )
    else:
        selected = empty_dataframe(SELECTED_VARIANT_COLUMNS)
        selected.to_csv(paths["selected_variants"], index=False)

    return {
        "tables": {
            "grid_validation": grid_results,
            "grid_ranking": ranking,
            "selected_variants": selected,
            "performance": final_performance,
            "failures": failures,
        },
        "paths": paths,
    }


def save_selection(
    *,
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Write ranking and selected-variant tables."""
    output = config["output"]
    ranking_path = Path(output["grid_ranking_summary"])
    selected_path = Path(output["selected_variants"])
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(ranking_path, index=False)
    selected.to_csv(selected_path, index=False)


def load_selected_variants(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate the frozen representative variants."""
    path = Path(config["output"]["selected_variants"])

    if not path.is_file():
        raise FileNotFoundError(
            "Selected-variant table not found. Run --stage select first: "
            f"{path}"
        )

    dataframe = pd.read_csv(path, keep_default_na=False)
    required = (
        "task_id",
        "model_id",
        "variant_id",
        "parameters_json",
        "selected",
    )
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Selected-variant table lacks columns: "
            + ", ".join(missing)
        )

    dataframe["selected"] = pd.to_numeric(
        dataframe["selected"], errors="raise"
    ).astype(int)
    dataframe = dataframe[dataframe["selected"] == 1].copy()

    if dataframe.duplicated(
        subset=["task_id", "model_id"],
        keep=False,
    ).any():
        raise ValueError(
            "More than one variant is selected for a task/model pair."
        )

    return dataframe


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    configuration_sha256: str,
    representation_manifest: dict[str, Any],
    partition_manifest: dict[str, Any],
    summaries: dict[str, Any],
    invocation: dict[str, Any],
) -> dict[str, Any]:
    """Build the global modelling manifest."""
    output_records: dict[str, Any] = {}

    for name, path in summaries["paths"].items():
        table = summaries["tables"][name]
        output_records[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
        }

    return {
        "schema_version": "2.0",
        "created_at_utc": utc_now(),
        "purpose": (
            "Fixed-grid model characterization followed by one global "
            "representative variant per task and model family."
        ),
        "configuration": config,
        "configuration_path": str(config_path),
        "configuration_sha256": configuration_sha256,
        "methodological_controls": {
            "per_split_hyperparameter_tuning": False,
            "per_split_threshold_tuning": False,
            "grid_uses_reference_regime_only": True,
            "grid_uses_train_and_validation_only": True,
            "grid_test_evaluations": 0,
            "representative_selection_metric": (
                "median validation average precision across reference seeds"
            ),
            "same_selected_variant_across_regimes": True,
            "test_used_for_variant_selection": False,
            "test_evaluated_once_per_final_split": True,
            "scaler_fitted_on_train_only": True,
            "refit_on_train_validation": False,
            "linear_svm_calibration": "none",
            "linear_svm_score_type": "decision_function",
        },
        "inputs": {
            "representation_manifest": {
                "path": config["input"]["representation_manifest"],
                "sha256": sha256_file(
                    Path(config["input"]["representation_manifest"])
                ),
            },
            "embeddings": {
                "path": config["input"]["embeddings"],
                "sha256": representation_manifest["output"]["sha256"],
                "rows": representation_manifest["output"]["rows"],
                "dimension": representation_manifest["representation"][
                    "output_dimension"
                ],
            },
            "partition_generation_manifest": {
                "path": config["input"][
                    "partition_generation_manifest"
                ],
                "sha256": sha256_file(
                    Path(
                        config["input"][
                            "partition_generation_manifest"
                        ]
                    )
                ),
            },
            "partition_index": {
                "path": config["input"]["partition_index"],
                "sha256": sha256_file(
                    Path(config["input"]["partition_index"])
                ),
            },
            "shared_task_dataset_sha256": representation_manifest[
                "input"
            ]["sha256"],
        },
        "invocation": invocation,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": package_version("scikit-learn"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "scipy": package_version("scipy"),
            "joblib": package_version("joblib"),
        },
        "partition_manifest_schema_version": partition_manifest.get(
            "schema_version"
        ),
        "outputs": output_records,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed model variants, select representative "
            "variants globally, and perform final held-out evaluation."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/modelling.yml.",
    )
    parser.add_argument(
        "--stage",
        choices=["grid", "select", "final", "all", "summarize"],
        default="all",
        help=(
            "grid: reference validation only; select: aggregate grid; "
            "final: selected variants on all regimes; all: complete flow."
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seeds or inclusive ranges, for example 0 10-19.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional maximum number of task/split/model runs.",
    )
    parser.add_argument(
        "--allow-partial-selection",
        action="store_true",
        help="Allow selection from fewer than the configured seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the execution plan only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching run directories.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested modelling stage."""
    args = parse_arguments()
    config_path = Path(args.config)

    try:
        print(f"train_amp_models.py version: {SCRIPT_VERSION}")

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        config = load_yaml(config_path)
        validate_configuration(config)
        verify_software(config)
        configuration_sha256 = sha256_file(config_path)
        representation_manifest, partition_manifest = (
            verify_input_manifests(config)
        )
        full_partition_index = load_partition_index(config)
        models = enabled_models(config, args.models)
        seeds = parse_integer_filter(args.seeds)
        reference_regime = str(
            config["selection"]["reference_regime"]
        )

        needs_embeddings = args.stage in {
            "grid",
            "final",
            "all",
        }
        embedding_matrix: np.ndarray | None = None
        embedding_index: pd.Index | None = None
        embedding_columns: list[str] = []

        if needs_embeddings and not args.dry_run:
            (
                embedding_matrix,
                embedding_index,
                embedding_columns,
            ) = load_embeddings(config, representation_manifest)
            print(
                "Loaded frozen embeddings: "
                f"{embedding_matrix.shape[0]} x "
                f"{embedding_matrix.shape[1]}"
            )

        completed = 0
        skipped = 0
        failed = 0

        if args.stage in {"grid", "all"}:
            grid_index = filter_partition_index(
                full_partition_index[
                    full_partition_index["regime_id"]
                    == reference_regime
                ].copy(),
                tasks=args.tasks,
                regimes=[reference_regime],
                seeds=seeds,
            )
            plan = [
                (row, model_id)
                for _, row in grid_index.iterrows()
                for model_id in models
            ]

            if args.max_runs is not None:
                if args.max_runs < 1:
                    raise ValueError("--max-runs must be positive.")
                plan = plan[: args.max_runs]

            print("Fixed-grid validation plan")
            print("--------------------------")
            print(f"Reference regime: {reference_regime}")
            print(f"Partition sets: {len(grid_index)}")
            print(f"Model-family runs: {len(plan)}")
            print("Test accessed during grid: no")
            print("Per-split hyperparameter selection: no")

            if not args.dry_run:
                assert embedding_matrix is not None
                assert embedding_index is not None

                for run_number, (index_row, model_id) in enumerate(
                    plan,
                    start=1,
                ):
                    print(
                        f"[grid {run_number}/{len(plan)}] "
                        f"{index_row['task_id']} / "
                        f"seed {int(index_row['seed'])} / {model_id}"
                    )

                    try:
                        result = run_grid_model(
                            index_row=index_row,
                            model_id=model_id,
                            model_config=models[model_id],
                            config=config,
                            configuration_sha256=configuration_sha256,
                            embedding_matrix=embedding_matrix,
                            embedding_index=embedding_index,
                            representation_manifest=representation_manifest,
                            overwrite=args.overwrite,
                        )

                        if result["status"] == "skipped_existing":
                            skipped += 1
                            print("  skipped: already complete")
                        else:
                            completed += 1
                            print("  complete")

                    except Exception as error:
                        failed += 1
                        print(
                            f"  failed: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )

                        if not config["execution"].get(
                            "continue_after_run_failure", True
                        ):
                            raise

        if args.stage in {"select", "all"} and not args.dry_run:
            run_root = Path(config["output"]["run_root"])
            grid_results, _ = collect_grid_results(
                run_root,
                configuration_sha256,
            )

            if args.tasks:
                grid_results = grid_results[
                    grid_results["task_id"].isin(args.tasks)
                ]

            if args.models:
                grid_results = grid_results[
                    grid_results["model_id"].isin(args.models)
                ]

            ranked, selected = select_representative_variants(
                grid_results=grid_results,
                config=config,
                allow_partial=args.allow_partial_selection,
            )
            save_selection(
                ranked=ranked,
                selected=selected,
                config=config,
            )
            print("Representative variants selected")
            print("--------------------------------")

            for row in selected.itertuples(index=False):
                print(
                    f"{row.task_id} / {row.model_id}: "
                    f"{row.variant_id} "
                    f"(median validation AP="
                    f"{row.validation_ap_median:.6f})"
                )

        if args.stage in {"final", "all"}:
            selected = load_selected_variants(config)

            if args.tasks:
                selected = selected[
                    selected["task_id"].isin(args.tasks)
                ]

            if args.models:
                selected = selected[
                    selected["model_id"].isin(args.models)
                ]

            final_index = filter_partition_index(
                full_partition_index,
                tasks=args.tasks,
                regimes=args.regimes,
                seeds=seeds,
            )
            selection_path = Path(config["output"]["selected_variants"])
            selection_hash = sha256_file(selection_path)
            selection_lookup = {
                (str(row["task_id"]), str(row["model_id"])): row
                for _, row in selected.iterrows()
            }
            plan = []

            for _, index_row in final_index.iterrows():
                for model_id in models:
                    key = (str(index_row["task_id"]), model_id)

                    if key not in selection_lookup:
                        continue

                    plan.append(
                        (
                            index_row,
                            model_id,
                            selection_lookup[key],
                        )
                    )

            if args.max_runs is not None:
                plan = plan[: args.max_runs]

            print("Final fixed-variant evaluation plan")
            print("-----------------------------------")
            print(f"Partition sets: {len(final_index)}")
            print(f"Final model runs: {len(plan)}")
            print("Per-split hyperparameter selection: no")
            print("Per-split threshold selection: no")
            print("Test scores per final run: one")

            if not args.dry_run:
                assert embedding_matrix is not None
                assert embedding_index is not None

                for run_number, (
                    index_row,
                    model_id,
                    selected_row,
                ) in enumerate(plan, start=1):
                    print(
                        f"[final {run_number}/{len(plan)}] "
                        f"{index_row['task_id']} / "
                        f"{index_row['regime_id']} / "
                        f"seed {int(index_row['seed'])} / "
                        f"{model_id}:{selected_row['variant_id']}"
                    )

                    try:
                        result = run_final_model(
                            index_row=index_row,
                            selected_row=selected_row,
                            model_id=model_id,
                            model_config=models[model_id],
                            config=config,
                            configuration_sha256=configuration_sha256,
                            selection_file_sha256=selection_hash,
                            embedding_matrix=embedding_matrix,
                            embedding_index=embedding_index,
                            representation_manifest=representation_manifest,
                            overwrite=args.overwrite,
                        )

                        if result["status"] == "skipped_existing":
                            skipped += 1
                            print("  skipped: already complete")
                        else:
                            completed += 1
                            test_ap = result["manifest"]["test_metrics"][
                                "average_precision"
                            ]
                            print(f"  complete: test AP={test_ap:.6f}")

                    except Exception as error:
                        failed += 1
                        print(
                            f"  failed: {type(error).__name__}: {error}",
                            file=sys.stderr,
                        )

                        if not config["execution"].get(
                            "continue_after_run_failure", True
                        ):
                            raise

        if args.dry_run:
            return 0

        summaries = write_summary_tables(
            config=config,
            configuration_sha256=configuration_sha256,
        )
        manifest = build_manifest(
            config=config,
            config_path=config_path,
            configuration_sha256=configuration_sha256,
            representation_manifest=representation_manifest,
            partition_manifest=partition_manifest,
            summaries=summaries,
            invocation={
                "stage": args.stage,
                "tasks": args.tasks,
                "regimes": args.regimes,
                "seeds": args.seeds,
                "models": args.models,
                "max_runs": args.max_runs,
                "allow_partial_selection": args.allow_partial_selection,
                "overwrite": args.overwrite,
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
                "embedding_dimension": len(embedding_columns),
            },
        )
        manifest_path = Path(config["output"]["manifest"])
        write_json(manifest, manifest_path)

        print()
        print("Modelling workflow completed")
        print("----------------------------")
        print(f"Completed now: {completed}")
        print(f"Skipped existing: {skipped}")
        print(f"Failed now: {failed}")
        print(
            "Grid validation summary: "
            f"{config['output']['grid_validation_summary']}"
        )
        print(
            "Selected variants: "
            f"{config['output']['selected_variants']}"
        )
        print(
            "Performance summary: "
            f"{config['output']['performance_summary']}"
        )
        print(f"Manifest: {manifest_path}")

        return 2 if failed > 0 else 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        ImportError,
        pd.errors.ParserError,
        yaml.YAMLError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
