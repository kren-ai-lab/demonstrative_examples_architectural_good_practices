#!/usr/bin/env python3

"""Audit homology configurations before constructing definitive splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def decode_separator(value: str) -> str:
    """Convert escaped separators such as '\\t' into actual characters."""
    decoded = bytes(value, "utf-8").decode("unicode_escape")

    if len(decoded) != 1:
        raise ValueError(
            "The separator must resolve to exactly one character."
        )

    return decoded


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a file SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the audit configuration."""
    required_sections = [
        "input",
        "dataset",
        "split_audit",
        "feasibility",
        "output",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(missing_sections)
        )

    proportions = config["split_audit"].get(
        "proportions",
        {},
    )

    if len(proportions) < 2:
        raise ValueError(
            "At least two split proportions must be configured."
        )

    proportion_sum = sum(
        float(value)
        for value in proportions.values()
    )

    if not math.isclose(
        proportion_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Split proportions must sum to 1.0. "
            f"Observed sum: {proportion_sum}"
        )

    for split_name, value in proportions.items():
        if float(value) <= 0:
            raise ValueError(
                f"Split proportion '{split_name}' must be positive."
            )

    number_of_seeds = int(
        config["split_audit"].get(
            "number_of_seeds",
            0,
        )
    )

    attempts_per_seed = int(
        config["split_audit"].get(
            "attempts_per_seed",
            0,
        )
    )

    if number_of_seeds < 1:
        raise ValueError(
            "split_audit.number_of_seeds must be positive."
        )

    if attempts_per_seed < 1:
        raise ValueError(
            "split_audit.attempts_per_seed must be positive."
        )


def load_task_dataset(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate the task dataset."""
    input_config = config["input"]
    dataset_config = config["dataset"]

    task_path = Path(input_config["task_dataset"])

    if not task_path.is_file():
        raise FileNotFoundError(
            f"Task dataset not found: {task_path}"
        )

    separator = decode_separator(
        str(input_config.get("separator", ","))
    )

    encoding = str(
        input_config.get("encoding", "utf-8")
    )

    dataframe = pd.read_csv(
        task_path,
        sep=separator,
        encoding=encoding,
        keep_default_na=False,
    )

    required_columns = [
        dataset_config["task_column"],
        dataset_config["sequence_id_column"],
        dataset_config["label_column"],
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required task-dataset columns were not found: "
            + ", ".join(missing_columns)
        )

    task_column = dataset_config["task_column"]
    sequence_id_column = dataset_config[
        "sequence_id_column"
    ]
    label_column = dataset_config["label_column"]

    dataframe[task_column] = (
        dataframe[task_column]
        .astype(str)
        .str.strip()
    )

    dataframe[sequence_id_column] = (
        dataframe[sequence_id_column]
        .astype(str)
        .str.strip()
    )

    dataframe[label_column] = pd.to_numeric(
        dataframe[label_column],
        errors="raise",
    ).astype(int)

    unexpected_labels = sorted(
        set(dataframe[label_column].unique()) - {0, 1}
    )

    if unexpected_labels:
        raise ValueError(
            "Only labels 0 and 1 are supported. Found: "
            + ", ".join(map(str, unexpected_labels))
        )

    included_tasks = dataset_config.get(
        "included_tasks",
    )

    if included_tasks:
        missing_tasks = sorted(
            set(included_tasks)
            - set(dataframe[task_column].unique())
        )

        if missing_tasks:
            raise ValueError(
                "Configured tasks were not found: "
                + ", ".join(missing_tasks)
            )

        dataframe = dataframe[
            dataframe[task_column].isin(included_tasks)
        ].copy()

    duplicate_mask = dataframe.duplicated(
        subset=[
            task_column,
            sequence_id_column,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicated task/sequence combinations were found."
        )

    return dataframe


def find_membership_files(
    config: dict[str, Any],
) -> list[tuple[str, str, Path]]:
    """Find configured cluster-membership files."""
    input_config = config["input"]
    dataset_config = config["dataset"]

    homology_root = Path(
        input_config["homology_root"]
    )

    if not homology_root.is_dir():
        raise FileNotFoundError(
            f"Homology output root not found: {homology_root}"
        )

    included_tasks = dataset_config.get(
        "included_tasks",
        [],
    )

    included_configurations = dataset_config.get(
        "included_configurations",
        [],
    )

    if not included_tasks:
        included_tasks = sorted(
            path.name
            for path in homology_root.iterdir()
            if path.is_dir()
        )

    files: list[tuple[str, str, Path]] = []

    for task_id in included_tasks:
        task_directory = homology_root / task_id

        if not task_directory.is_dir():
            raise FileNotFoundError(
                f"Task homology directory not found: {task_directory}"
            )

        if included_configurations:
            configuration_ids = included_configurations
        else:
            configuration_ids = sorted(
                path.name
                for path in task_directory.iterdir()
                if path.is_dir()
            )

        for configuration_id in configuration_ids:
            membership_path = (
                task_directory
                / configuration_id
                / "cluster_membership.csv"
            )

            if not membership_path.is_file():
                raise FileNotFoundError(
                    "Cluster-membership file not found: "
                    f"{membership_path}"
                )

            files.append(
                (
                    str(task_id),
                    str(configuration_id),
                    membership_path,
                )
            )

    return files


def load_and_validate_membership(
    membership_path: Path,
    task_id: str,
    configuration_id: str,
    task_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate one clustering membership table."""
    dataset_config = config["dataset"]

    task_column = dataset_config["task_column"]
    sequence_id_column = dataset_config[
        "sequence_id_column"
    ]
    label_column = dataset_config["label_column"]

    membership = pd.read_csv(
        membership_path,
        keep_default_na=False,
    )

    required_columns = [
        "task_id",
        "homology_config_id",
        "sequence_id",
        "cluster_id",
        "amp_label",
        "minimum_sequence_identity",
        "minimum_coverage",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in membership.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {membership_path}: "
            + ", ".join(missing_columns)
        )

    if set(membership["task_id"].astype(str)) != {task_id}:
        raise ValueError(
            f"Unexpected task IDs in {membership_path}"
        )

    if (
        set(membership["homology_config_id"].astype(str))
        != {configuration_id}
    ):
        raise ValueError(
            f"Unexpected configuration IDs in {membership_path}"
        )

    if membership["sequence_id"].duplicated().any():
        raise ValueError(
            "A sequence appears more than once in "
            f"{membership_path}"
        )

    task_subset = task_dataframe[
        task_dataframe[task_column] == task_id
    ][
        [
            sequence_id_column,
            label_column,
        ]
    ].copy()

    expected_ids = set(
        task_subset[sequence_id_column].astype(str)
    )

    observed_ids = set(
        membership["sequence_id"].astype(str)
    )

    if expected_ids != observed_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)

        message_parts = []

        if missing:
            message_parts.append(
                "missing: " + ", ".join(missing[:10])
            )

        if extra:
            message_parts.append(
                "unexpected: " + ", ".join(extra[:10])
            )

        raise ValueError(
            f"Membership mismatch in {membership_path}: "
            + "; ".join(message_parts)
        )

    expected_labels = task_subset.rename(
        columns={
            sequence_id_column: "sequence_id",
            label_column: "expected_label",
        }
    )

    membership = membership.merge(
        expected_labels,
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )

    membership["amp_label"] = pd.to_numeric(
        membership["amp_label"],
        errors="raise",
    ).astype(int)

    label_mismatch = (
        membership["amp_label"]
        != membership["expected_label"]
    )

    if label_mismatch.any():
        raise ValueError(
            "Labels in the membership table do not match "
            f"the task dataset: {membership_path}"
        )

    return membership.drop(
        columns=["expected_label"]
    )


def build_cluster_table(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per indivisible homology group."""
    cluster_table = (
        membership.groupby(
            "cluster_id",
            observed=True,
            sort=True,
        )
        .agg(
            total_count=("sequence_id", "size"),
            positive_count=("amp_label", "sum"),
        )
        .reset_index()
    )

    cluster_table["negative_count"] = (
        cluster_table["total_count"]
        - cluster_table["positive_count"]
    )

    cluster_table["mixed_label_cluster"] = (
        (cluster_table["positive_count"] > 0)
        & (cluster_table["negative_count"] > 0)
    ).astype(int)

    return cluster_table


def assignment_objective(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
    weights: dict[str, float],
    overflow_penalty: float,
) -> float:
    """Calculate the global assignment objective."""
    score = 0.0

    components = [
        ("total_count", "total_size"),
        ("positive_count", "positive_count"),
        ("negative_count", "negative_count"),
    ]

    for split_name in counts:
        for count_name, weight_name in components:
            target = max(
                targets[split_name][count_name],
                1.0,
            )

            observed = counts[split_name][count_name]

            deviation = abs(
                observed - target
            ) / target

            score += (
                float(weights.get(weight_name, 1.0))
                * deviation
            )

            if observed > target:
                overflow = (
                    observed - target
                ) / target

                score += (
                    overflow_penalty
                    * float(weights.get(weight_name, 1.0))
                    * overflow
                )

    return score


def calculate_assignment_metrics(
    assignment: pd.DataFrame,
    proportions: dict[str, float],
    global_prevalence: float,
) -> dict[str, Any]:
    """Calculate split-level and aggregate feasibility metrics."""
    total_sequences = int(
        assignment["total_count"].sum()
    )

    metrics: dict[str, Any] = {}

    maximum_size_deviation = 0.0
    maximum_prevalence_deviation = 0.0

    minimum_sequences = math.inf
    minimum_positives = math.inf
    minimum_negatives = math.inf

    for split_name, target_fraction in proportions.items():
        split_clusters = assignment[
            assignment["partition"] == split_name
        ]

        split_total = int(
            split_clusters["total_count"].sum()
        )
        split_positive = int(
            split_clusters["positive_count"].sum()
        )
        split_negative = int(
            split_clusters["negative_count"].sum()
        )

        split_prevalence = (
            split_positive / split_total
            if split_total > 0
            else math.nan
        )

        target_total = (
            total_sequences * float(target_fraction)
        )

        size_deviation = (
            abs(split_total - target_total)
            / max(target_total, 1.0)
        )

        prevalence_deviation = (
            abs(split_prevalence - global_prevalence)
            if split_total > 0
            else math.inf
        )

        metrics[f"{split_name}_sequences"] = split_total
        metrics[f"{split_name}_positives"] = split_positive
        metrics[f"{split_name}_negatives"] = split_negative
        metrics[f"{split_name}_prevalence"] = split_prevalence
        metrics[
            f"{split_name}_size_deviation"
        ] = size_deviation
        metrics[
            f"{split_name}_prevalence_deviation"
        ] = prevalence_deviation
        metrics[
            f"{split_name}_cluster_count"
        ] = int(len(split_clusters))

        maximum_size_deviation = max(
            maximum_size_deviation,
            size_deviation,
        )

        maximum_prevalence_deviation = max(
            maximum_prevalence_deviation,
            prevalence_deviation,
        )

        minimum_sequences = min(
            minimum_sequences,
            split_total,
        )
        minimum_positives = min(
            minimum_positives,
            split_positive,
        )
        minimum_negatives = min(
            minimum_negatives,
            split_negative,
        )

    metrics["maximum_split_size_deviation"] = (
        maximum_size_deviation
    )
    metrics["maximum_prevalence_deviation"] = (
        maximum_prevalence_deviation
    )
    metrics["minimum_split_sequences"] = int(
        minimum_sequences
    )
    metrics["minimum_split_positives"] = int(
        minimum_positives
    )
    metrics["minimum_split_negatives"] = int(
        minimum_negatives
    )

    return metrics


def is_feasible(
    metrics: dict[str, Any],
    feasibility_config: dict[str, Any],
) -> bool:
    """Evaluate whether a diagnostic assignment is feasible."""
    return all(
        [
            metrics["maximum_split_size_deviation"]
            <= float(
                feasibility_config[
                    "maximum_split_size_deviation"
                ]
            ),
            metrics["maximum_prevalence_deviation"]
            <= float(
                feasibility_config[
                    "maximum_prevalence_deviation"
                ]
            ),
            metrics["minimum_split_sequences"]
            >= int(
                feasibility_config[
                    "minimum_sequences_per_split"
                ]
            ),
            metrics["minimum_split_positives"]
            >= int(
                feasibility_config[
                    "minimum_positive_sequences_per_split"
                ]
            ),
            metrics["minimum_split_negatives"]
            >= int(
                feasibility_config[
                    "minimum_negative_sequences_per_split"
                ]
            ),
        ]
    )


def assign_clusters_once(
    cluster_table: pd.DataFrame,
    proportions: dict[str, float],
    weights: dict[str, float],
    overflow_penalty: float,
    random_generator: random.Random,
) -> tuple[pd.DataFrame, float]:
    """Assign complete clusters to partitions using a greedy objective."""
    split_names = list(proportions)

    total_sequences = int(
        cluster_table["total_count"].sum()
    )
    total_positives = int(
        cluster_table["positive_count"].sum()
    )
    total_negatives = int(
        cluster_table["negative_count"].sum()
    )

    targets = {
        split_name: {
            "total_count": (
                total_sequences * proportions[split_name]
            ),
            "positive_count": (
                total_positives * proportions[split_name]
            ),
            "negative_count": (
                total_negatives * proportions[split_name]
            ),
        }
        for split_name in split_names
    }

    counts = {
        split_name: {
            "total_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        }
        for split_name in split_names
    }

    records = cluster_table.to_dict(
        orient="records"
    )

    random_generator.shuffle(records)

    # Los grupos grandes se asignan primero. El shuffle previo
    # aleatoriza los empates de tamaño.
    records.sort(
        key=lambda record: (
            record["total_count"],
            max(
                record["positive_count"],
                record["negative_count"],
            ),
        ),
        reverse=True,
    )

    assignments: list[dict[str, Any]] = []

    for cluster in records:
        candidate_scores: list[
            tuple[float, float, str]
        ] = []

        for split_name in split_names:
            candidate_counts = {
                name: values.copy()
                for name, values in counts.items()
            }

            candidate_counts[split_name][
                "total_count"
            ] += int(cluster["total_count"])

            candidate_counts[split_name][
                "positive_count"
            ] += int(cluster["positive_count"])

            candidate_counts[split_name][
                "negative_count"
            ] += int(cluster["negative_count"])

            score = assignment_objective(
                counts=candidate_counts,
                targets=targets,
                weights=weights,
                overflow_penalty=overflow_penalty,
            )

            candidate_scores.append(
                (
                    score,
                    random_generator.random(),
                    split_name,
                )
            )

        candidate_scores.sort()
        selected_split = candidate_scores[0][2]

        counts[selected_split]["total_count"] += int(
            cluster["total_count"]
        )
        counts[selected_split]["positive_count"] += int(
            cluster["positive_count"]
        )
        counts[selected_split]["negative_count"] += int(
            cluster["negative_count"]
        )

        assignments.append(
            {
                **cluster,
                "partition": selected_split,
            }
        )

    assignment = pd.DataFrame(assignments)

    final_objective = assignment_objective(
        counts=counts,
        targets=targets,
        weights=weights,
        overflow_penalty=overflow_penalty,
    )

    return assignment, final_objective


def audit_one_configuration(
    membership: pd.DataFrame,
    task_id: str,
    configuration_id: str,
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Audit one task and homology configuration."""
    split_config = config["split_audit"]
    feasibility_config = config["feasibility"]

    proportions = {
        str(name): float(value)
        for name, value
        in split_config["proportions"].items()
    }

    weights = {
        str(name): float(value)
        for name, value
        in split_config.get(
            "objective_weights",
            {},
        ).items()
    }

    overflow_penalty = float(
        split_config.get(
            "overflow_penalty",
            2.0,
        )
    )

    number_of_seeds = int(
        split_config["number_of_seeds"]
    )
    first_seed = int(
        split_config.get("first_seed", 0)
    )
    attempts_per_seed = int(
        split_config["attempts_per_seed"]
    )

    cluster_table = build_cluster_table(
        membership
    )

    total_sequences = int(
        len(membership)
    )
    total_positives = int(
        membership["amp_label"].sum()
    )
    total_negatives = (
        total_sequences - total_positives
    )

    global_prevalence = (
        total_positives / total_sequences
    )

    cluster_sizes = cluster_table[
        "total_count"
    ]

    largest_cluster_size = int(
        cluster_sizes.max()
    )
    largest_cluster_fraction = (
        largest_cluster_size / total_sequences
    )

    threshold = float(
        membership[
            "minimum_sequence_identity"
        ].iloc[0]
    )

    minimum_coverage = float(
        membership[
            "minimum_coverage"
        ].iloc[0]
    )

    detailed_runs: list[dict[str, Any]] = []

    best_global_assignment: pd.DataFrame | None = None
    best_global_record: dict[str, Any] | None = None

    for seed_offset in range(number_of_seeds):
        seed = first_seed + seed_offset

        best_seed_assignment: pd.DataFrame | None = None
        best_seed_record: dict[str, Any] | None = None

        for attempt in range(attempts_per_seed):
            derived_seed = (
                seed * 1_000_003
                + attempt
            )

            rng = random.Random(
                derived_seed
            )

            assignment, objective = (
                assign_clusters_once(
                    cluster_table=cluster_table,
                    proportions=proportions,
                    weights=weights,
                    overflow_penalty=overflow_penalty,
                    random_generator=rng,
                )
            )

            metrics = (
                calculate_assignment_metrics(
                    assignment=assignment,
                    proportions=proportions,
                    global_prevalence=(
                        global_prevalence
                    ),
                )
            )

            feasible = is_feasible(
                metrics=metrics,
                feasibility_config=(
                    feasibility_config
                ),
            )

            candidate_record = {
                "task_id": task_id,
                "homology_config_id": (
                    configuration_id
                ),
                "minimum_sequence_identity": threshold,
                "minimum_coverage": minimum_coverage,
                "seed": seed,
                "attempt": attempt,
                "derived_seed": derived_seed,
                "objective": objective,
                "feasible": int(feasible),
                **metrics,
            }

            if best_seed_record is None:
                best_seed_record = candidate_record
                best_seed_assignment = assignment
                continue

            current_key = (
                -candidate_record["feasible"],
                candidate_record["objective"],
            )
            best_key = (
                -best_seed_record["feasible"],
                best_seed_record["objective"],
            )

            if current_key < best_key:
                best_seed_record = candidate_record
                best_seed_assignment = assignment

        if (
            best_seed_record is None
            or best_seed_assignment is None
        ):
            raise RuntimeError(
                "No diagnostic assignments were generated."
            )

        detailed_runs.append(
            best_seed_record
        )

        if best_global_record is None:
            best_global_record = best_seed_record
            best_global_assignment = (
                best_seed_assignment
            )
        else:
            current_key = (
                -best_seed_record["feasible"],
                best_seed_record["objective"],
            )
            best_key = (
                -best_global_record["feasible"],
                best_global_record["objective"],
            )

            if current_key < best_key:
                best_global_record = (
                    best_seed_record
                )
                best_global_assignment = (
                    best_seed_assignment
                )

    if (
        best_global_record is None
        or best_global_assignment is None
    ):
        raise RuntimeError(
            "No global diagnostic assignment was selected."
        )

    feasible_seeds = sum(
        record["feasible"]
        for record in detailed_runs
    )

    feasibility_rate = (
        feasible_seeds / number_of_seeds
    )

    structurally_acceptable = (
        largest_cluster_fraction
        <= float(
            feasibility_config[
                "maximum_largest_cluster_fraction"
            ]
        )
    )

    stable_feasibility = (
        feasibility_rate
        >= float(
            feasibility_config[
                "minimum_feasibility_rate"
            ]
        )
    )

    if (
        structurally_acceptable
        and stable_feasibility
    ):
        selection_status = "eligible"
    elif not structurally_acceptable:
        selection_status = (
            "structurally_problematic"
        )
    else:
        selection_status = (
            "split_feasibility_unstable"
        )

    mixed_clusters = int(
        cluster_table[
            "mixed_label_cluster"
        ].sum()
    )

    sequences_in_mixed_clusters = int(
        cluster_table.loc[
            cluster_table[
                "mixed_label_cluster"
            ]
            == 1,
            "total_count",
        ].sum()
    )

    summary = {
        "task_id": task_id,
        "homology_config_id": (
            configuration_id
        ),
        "minimum_sequence_identity": threshold,
        "minimum_coverage": minimum_coverage,
        "number_of_sequences": total_sequences,
        "positive_sequences": total_positives,
        "negative_sequences": total_negatives,
        "positive_prevalence": global_prevalence,
        "number_of_clusters": int(
            len(cluster_table)
        ),
        "number_of_singletons": int(
            (cluster_sizes == 1).sum()
        ),
        "singleton_fraction": float(
            (cluster_sizes == 1).mean()
        ),
        "largest_cluster_size": (
            largest_cluster_size
        ),
        "largest_cluster_fraction": (
            largest_cluster_fraction
        ),
        "mean_cluster_size": float(
            cluster_sizes.mean()
        ),
        "median_cluster_size": float(
            cluster_sizes.median()
        ),
        "number_of_mixed_label_clusters": (
            mixed_clusters
        ),
        "sequences_in_mixed_label_clusters": (
            sequences_in_mixed_clusters
        ),
        "mixed_cluster_sequence_fraction": (
            sequences_in_mixed_clusters
            / total_sequences
        ),
        "feasible_seeds": feasible_seeds,
        "number_of_seeds": number_of_seeds,
        "feasibility_rate": feasibility_rate,
        "best_seed": int(
            best_global_record["seed"]
        ),
        "best_attempt": int(
            best_global_record["attempt"]
        ),
        "best_objective": float(
            best_global_record["objective"]
        ),
        "best_maximum_split_size_deviation": (
            best_global_record[
                "maximum_split_size_deviation"
            ]
        ),
        "best_maximum_prevalence_deviation": (
            best_global_record[
                "maximum_prevalence_deviation"
            ]
        ),
        "best_minimum_split_sequences": (
            best_global_record[
                "minimum_split_sequences"
            ]
        ),
        "best_minimum_split_positives": (
            best_global_record[
                "minimum_split_positives"
            ]
        ),
        "best_minimum_split_negatives": (
            best_global_record[
                "minimum_split_negatives"
            ]
        ),
        "cluster_cross_partition_violations": 0,
        "exact_maximum_cross_split_identity": None,
        "cross_split_similarity_assessment": (
            "not_computed_during_structural_audit"
        ),
        "operational_identity_threshold": threshold,
        "selection_status": selection_status,
    }

    best_global_assignment = (
        best_global_assignment.copy()
    )

    best_global_assignment.insert(
        0,
        "task_id",
        task_id,
    )
    best_global_assignment.insert(
        1,
        "homology_config_id",
        configuration_id,
    )
    best_global_assignment.insert(
        2,
        "seed",
        int(best_global_record["seed"]),
    )

    return (
        summary,
        detailed_runs,
        best_global_assignment,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit MMseqs2 homology configurations using "
            "cluster structure and diagnostic split feasibility."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/homology_audit.yml.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing audit outputs.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the homology-threshold audit."""
    args = parse_arguments()

    config_path = Path(args.config)

    try:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        config = load_yaml(
            config_path
        )
        validate_config(
            config
        )

        output_config = config["output"]

        summary_path = Path(
            output_config["audit_summary"]
        )
        runs_path = Path(
            output_config["feasibility_runs"]
        )
        candidate_root = Path(
            output_config["candidate_root"]
        )
        manifest_path = Path(
            output_config["manifest"]
        )

        principal_outputs = [
            summary_path,
            runs_path,
            manifest_path,
        ]

        existing_outputs = [
            path
            for path in principal_outputs
            if path.exists()
        ]

        if existing_outputs and not args.overwrite:
            raise FileExistsError(
                "Audit outputs already exist: "
                + ", ".join(
                    str(path)
                    for path in existing_outputs
                )
                + ". Use --overwrite to replace them."
            )

        if args.overwrite:
            for path in principal_outputs:
                if path.exists():
                    path.unlink()

        task_dataframe = load_task_dataset(
            config
        )

        membership_files = (
            find_membership_files(
                config
            )
        )

        audit_summaries: list[
            dict[str, Any]
        ] = []

        feasibility_runs: list[
            dict[str, Any]
        ] = []

        manifest_inputs: list[
            dict[str, Any]
        ] = []

        for (
            task_id,
            configuration_id,
            membership_path,
        ) in membership_files:
            print(
                f"Auditing {task_id} / "
                f"{configuration_id}"
            )

            membership = (
                load_and_validate_membership(
                    membership_path=membership_path,
                    task_id=task_id,
                    configuration_id=(
                        configuration_id
                    ),
                    task_dataframe=task_dataframe,
                    config=config,
                )
            )

            (
                summary,
                detailed_runs,
                best_assignment,
            ) = audit_one_configuration(
                membership=membership,
                task_id=task_id,
                configuration_id=(
                    configuration_id
                ),
                config=config,
            )

            audit_summaries.append(
                summary
            )
            feasibility_runs.extend(
                detailed_runs
            )

            manifest_inputs.append(
                {
                    "task_id": task_id,
                    "homology_config_id": (
                        configuration_id
                    ),
                    "path": str(
                        membership_path
                    ),
                    "sha256": sha256_file(
                        membership_path
                    ),
                }
            )

            if output_config.get(
                "write_best_candidate_assignments",
                True,
            ):
                candidate_directory = (
                    candidate_root
                    / task_id
                    / configuration_id
                )

                candidate_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                candidate_path = (
                    candidate_directory
                    / "best_diagnostic_assignment.csv"
                )

                if (
                    candidate_path.exists()
                    and not args.overwrite
                ):
                    raise FileExistsError(
                        f"Candidate output exists: "
                        f"{candidate_path}"
                    )

                best_assignment.to_csv(
                    candidate_path,
                    index=False,
                )

            print(
                "  "
                f"clusters={summary['number_of_clusters']} "
                f"largest={summary['largest_cluster_size']} "
                f"singletons={summary['number_of_singletons']} "
                f"feasibility="
                f"{summary['feasibility_rate']:.2f} "
                f"status={summary['selection_status']}"
            )

        summary_dataframe = pd.DataFrame(
            audit_summaries
        )

        summary_dataframe[
            "strictness_rank"
        ] = summary_dataframe.groupby(
            "task_id"
        )[
            "minimum_sequence_identity"
        ].rank(
            method="dense",
            ascending=True,
        ).astype(int)

        summary_dataframe = (
            summary_dataframe.sort_values(
                [
                    "task_id",
                    "minimum_sequence_identity",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        runs_dataframe = pd.DataFrame(
            feasibility_runs
        ).sort_values(
            [
                "task_id",
                "homology_config_id",
                "seed",
            ]
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        runs_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_dataframe.to_csv(
            summary_path,
            index=False,
        )

        runs_dataframe.to_csv(
            runs_path,
            index=False,
        )

        task_dataset_path = Path(
            config["input"]["task_dataset"]
        )

        manifest = {
            "schema_version": "1.0",
            "created_at_utc": utc_now(),
            "purpose": (
                "Structural audit of MMseqs2 sequence-similarity "
                "configurations before predictive modelling."
            ),
            "selection_rule": (
                "Configurations are evaluated using cluster structure "
                "and split feasibility only. Predictive performance "
                "is not used."
            ),
            "task_dataset": {
                "path": str(
                    task_dataset_path
                ),
                "sha256": sha256_file(
                    task_dataset_path
                ),
            },
            "membership_inputs": (
                manifest_inputs
            ),
            "configuration": config,
            "outputs": {
                "audit_summary": {
                    "path": str(
                        summary_path
                    ),
                    "sha256": sha256_file(
                        summary_path
                    ),
                },
                "feasibility_runs": {
                    "path": str(
                        runs_path
                    ),
                    "sha256": sha256_file(
                        runs_path
                    ),
                },
            },
            "cross_split_similarity_note": (
                "This audit verifies that complete MMseqs2 groups can "
                "be assigned as indivisible units. Exact maximum "
                "cross-split sequence identity is not calculated here "
                "and must be evaluated after definitive splits are "
                "constructed."
            ),
        }

        write_json(
            manifest,
            manifest_path,
        )

        print()
        print("Homology audit completed")
        print("------------------------")
        print(
            f"Configurations audited: "
            f"{len(summary_dataframe)}"
        )
        print(
            f"Audit summary: {summary_path}"
        )
        print(
            f"Feasibility runs: {runs_path}"
        )
        print(
            f"Manifest: {manifest_path}"
        )

        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        pd.errors.ParserError,
        yaml.YAMLError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())