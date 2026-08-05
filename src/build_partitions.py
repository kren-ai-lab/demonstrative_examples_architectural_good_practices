#!/usr/bin/env python3

"""Build reproducible random-stratified and MMseqs2 group-aware partitions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_separator(value: str) -> str:
    decoded = bytes(value, "utf-8").decode("unicode_escape")
    if len(decoded) != 1:
        raise ValueError("The separator must resolve to exactly one character.")
    return decoded


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")
    return config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_hash(dataframe: pd.DataFrame) -> str:
    ordered = dataframe[["sequence_id", "partition"]].sort_values("sequence_id")
    payload = "\n".join(
        f"{sequence_id}\t{partition}"
        for sequence_id, partition in ordered.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_split_id(task_id: str, regime_id: str, seed: int, digest: str) -> str:
    payload = f"{task_id}\n{regime_id}\n{seed}\n{digest}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def validate_config(config: dict[str, Any]) -> None:
    for section in ("input", "dataset", "partitioning", "validation", "output"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")

    proportions = config["partitioning"].get("proportions", {})
    if set(proportions) != {"train", "validation", "test"}:
        raise ValueError("partitioning.proportions must define train, validation, and test.")

    total = sum(float(value) for value in proportions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split proportions must sum to 1.0; observed {total}.")

    if int(config["partitioning"].get("number_of_seeds", 0)) < 1:
        raise ValueError("partitioning.number_of_seeds must be positive.")

    if int(config["partitioning"].get("attempts_per_seed", 0)) < 1:
        raise ValueError("partitioning.attempts_per_seed must be positive.")

    output_format = str(config["output"].get("assignment_format", "csv.gz"))
    if output_format not in {"csv", "csv.gz"}:
        raise ValueError("output.assignment_format must be 'csv' or 'csv.gz'.")


def load_task_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    input_config = config["input"]
    dataset_config = config["dataset"]

    path = Path(input_config["task_dataset"])
    if not path.is_file():
        raise FileNotFoundError(f"Task dataset not found: {path}")

    dataframe = pd.read_csv(
        path,
        sep=decode_separator(str(input_config.get("separator", ","))),
        encoding=str(input_config.get("encoding", "utf-8")),
        keep_default_na=False,
    )

    required = [
        dataset_config["task_column"],
        dataset_config["sequence_id_column"],
        dataset_config["label_column"],
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError("Required task columns are missing: " + ", ".join(missing))

    task_column = dataset_config["task_column"]
    sequence_id_column = dataset_config["sequence_id_column"]
    label_column = dataset_config["label_column"]

    dataframe[task_column] = dataframe[task_column].astype(str).str.strip()
    dataframe[sequence_id_column] = dataframe[sequence_id_column].astype(str).str.strip()
    dataframe[label_column] = pd.to_numeric(dataframe[label_column], errors="raise").astype(int)

    if set(dataframe[label_column].unique()) - {0, 1}:
        raise ValueError("Only binary labels 0 and 1 are supported.")

    included_tasks = dataset_config.get("included_tasks")
    if included_tasks:
        missing_tasks = sorted(set(included_tasks) - set(dataframe[task_column].unique()))
        if missing_tasks:
            raise ValueError("Configured tasks were not found: " + ", ".join(missing_tasks))
        dataframe = dataframe[dataframe[task_column].isin(included_tasks)].copy()

    if dataframe.duplicated([task_column, sequence_id_column]).any():
        raise ValueError("Duplicated task/sequence combinations were found.")

    class_counts = dataframe.groupby(task_column)[label_column].nunique()
    incomplete = class_counts[class_counts < 2]
    if not incomplete.empty:
        raise ValueError(
            "The following tasks do not contain both classes: "
            + ", ".join(incomplete.index.astype(str))
        )

    return dataframe, path


def load_audit_summary(config: dict[str, Any]) -> pd.DataFrame | None:
    path_value = config["input"].get("homology_audit_summary")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Homology audit summary not found: {path}")
    return pd.read_csv(path, keep_default_na=False)


def load_audit_runs(config: dict[str, Any]) -> pd.DataFrame | None:
    path_value = config["input"].get("homology_feasibility_runs")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Homology feasibility runs not found: {path}")
    return pd.read_csv(path, keep_default_na=False)


def largest_remainder_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    raw = {name: total * float(value) for name, value in proportions.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def build_random_stratified_assignment(
    task_dataframe: pd.DataFrame,
    sequence_id_column: str,
    label_column: str,
    proportions: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Create exact-size partitions with approximately preserved prevalence."""
    rng = random.Random(seed)
    partition_order = ("train", "validation", "test")

    total_counts = largest_remainder_counts(len(task_dataframe), proportions)
    positive_total = int((task_dataframe[label_column] == 1).sum())
    positive_counts = largest_remainder_counts(positive_total, proportions)

    if any(positive_counts[name] > total_counts[name] for name in partition_order):
        raise ValueError(
            "The requested split proportions cannot preserve both classes "
            "with the available sample counts."
        )

    class_partition_counts = {
        1: positive_counts,
        0: {
            name: total_counts[name] - positive_counts[name]
            for name in partition_order
        },
    }

    records: list[dict[str, Any]] = []
    for label in (0, 1):
        ids = sorted(
            task_dataframe.loc[task_dataframe[label_column] == label, sequence_id_column]
            .astype(str)
            .tolist()
        )
        rng.shuffle(ids)
        cursor = 0
        for partition in partition_order:
            stop = cursor + class_partition_counts[label][partition]
            for sequence_id in ids[cursor:stop]:
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "amp_label": label,
                        "partition": partition,
                        "group_id": sequence_id,
                    }
                )
            cursor = stop

        if cursor != len(ids):
            raise RuntimeError(
                f"Random stratified allocation did not consume all class-{label} sequences."
            )

    return pd.DataFrame(records).sort_values("sequence_id").reset_index(drop=True)


def build_cluster_table(membership: pd.DataFrame) -> pd.DataFrame:
    cluster_table = (
        membership.groupby("cluster_id", observed=True, sort=True)
        .agg(total_count=("sequence_id", "size"), positive_count=("amp_label", "sum"))
        .reset_index()
    )
    cluster_table["negative_count"] = (
        cluster_table["total_count"] - cluster_table["positive_count"]
    )
    return cluster_table


def assignment_objective(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
    weights: dict[str, float],
    overflow_penalty: float,
) -> float:
    score = 0.0
    components = (
        ("total_count", "total_size"),
        ("positive_count", "positive_count"),
        ("negative_count", "negative_count"),
    )
    for partition in counts:
        for count_name, weight_name in components:
            target = max(targets[partition][count_name], 1.0)
            observed = counts[partition][count_name]
            deviation = abs(observed - target) / target
            weight = float(weights.get(weight_name, 1.0))
            score += weight * deviation
            if observed > target:
                score += overflow_penalty * weight * ((observed - target) / target)
    return score


def assign_clusters_once(
    cluster_table: pd.DataFrame,
    proportions: dict[str, float],
    weights: dict[str, float],
    overflow_penalty: float,
    derived_seed: int,
) -> tuple[pd.DataFrame, float]:
    rng = random.Random(derived_seed)
    partitions = list(proportions)

    totals = {
        "total_count": int(cluster_table["total_count"].sum()),
        "positive_count": int(cluster_table["positive_count"].sum()),
        "negative_count": int(cluster_table["negative_count"].sum()),
    }
    targets = {
        partition: {
            component: totals[component] * proportions[partition]
            for component in totals
        }
        for partition in partitions
    }
    counts = {
        partition: {component: 0 for component in totals}
        for partition in partitions
    }

    records = cluster_table.to_dict(orient="records")
    rng.shuffle(records)
    records.sort(
        key=lambda row: (
            row["total_count"],
            max(row["positive_count"], row["negative_count"]),
        ),
        reverse=True,
    )

    assignments: list[dict[str, Any]] = []
    for cluster in records:
        candidates: list[tuple[float, float, str]] = []
        for partition in partitions:
            candidate_counts = {
                name: values.copy() for name, values in counts.items()
            }
            for component in totals:
                candidate_counts[partition][component] += int(cluster[component])
            score = assignment_objective(
                candidate_counts,
                targets,
                weights,
                overflow_penalty,
            )
            candidates.append((score, rng.random(), partition))

        candidates.sort()
        selected = candidates[0][2]
        for component in totals:
            counts[selected][component] += int(cluster[component])
        assignments.append({**cluster, "partition": selected})

    assignment = pd.DataFrame(assignments)
    objective = assignment_objective(counts, targets, weights, overflow_penalty)
    return assignment, objective


def assignment_metrics(
    cluster_assignment: pd.DataFrame,
    proportions: dict[str, float],
    global_prevalence: float,
) -> dict[str, Any]:
    total_sequences = int(cluster_assignment["total_count"].sum())
    metrics: dict[str, Any] = {}
    max_size_deviation = 0.0
    max_prevalence_deviation = 0.0
    minimum_sequences = math.inf
    minimum_positives = math.inf
    minimum_negatives = math.inf

    for partition, fraction in proportions.items():
        subset = cluster_assignment[cluster_assignment["partition"] == partition]
        total = int(subset["total_count"].sum())
        positives = int(subset["positive_count"].sum())
        negatives = int(subset["negative_count"].sum())
        prevalence = positives / total if total else math.nan
        target = total_sequences * fraction
        size_deviation = abs(total - target) / max(target, 1.0)
        prevalence_deviation = (
            abs(prevalence - global_prevalence) if total else math.inf
        )

        metrics[f"{partition}_sequences"] = total
        metrics[f"{partition}_positives"] = positives
        metrics[f"{partition}_negatives"] = negatives
        metrics[f"{partition}_prevalence"] = prevalence
        metrics[f"{partition}_size_deviation"] = size_deviation
        metrics[f"{partition}_prevalence_deviation"] = prevalence_deviation
        metrics[f"{partition}_group_count"] = int(len(subset))

        max_size_deviation = max(max_size_deviation, size_deviation)
        max_prevalence_deviation = max(max_prevalence_deviation, prevalence_deviation)
        minimum_sequences = min(minimum_sequences, total)
        minimum_positives = min(minimum_positives, positives)
        minimum_negatives = min(minimum_negatives, negatives)

    metrics["maximum_split_size_deviation"] = max_size_deviation
    metrics["maximum_prevalence_deviation"] = max_prevalence_deviation
    metrics["minimum_split_sequences"] = int(minimum_sequences)
    metrics["minimum_split_positives"] = int(minimum_positives)
    metrics["minimum_split_negatives"] = int(minimum_negatives)
    return metrics


def metrics_are_feasible(metrics: dict[str, Any], validation: dict[str, Any]) -> bool:
    return all(
        [
            metrics["maximum_split_size_deviation"]
            <= float(validation["maximum_split_size_deviation"]),
            metrics["maximum_prevalence_deviation"]
            <= float(validation["maximum_prevalence_deviation"]),
            metrics["minimum_split_sequences"]
            >= int(validation["minimum_sequences_per_split"]),
            metrics["minimum_split_positives"]
            >= int(validation["minimum_positive_sequences_per_split"]),
            metrics["minimum_split_negatives"]
            >= int(validation["minimum_negative_sequences_per_split"]),
        ]
    )


def select_group_assignment(
    cluster_table: pd.DataFrame,
    proportions: dict[str, float],
    weights: dict[str, float],
    overflow_penalty: float,
    seed: int,
    attempts_per_seed: int,
    validation: dict[str, Any],
    preferred_attempt: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global_prevalence = float(
        cluster_table["positive_count"].sum() / cluster_table["total_count"].sum()
    )

    attempts = (
        [preferred_attempt]
        if preferred_attempt is not None
        else list(range(attempts_per_seed))
    )

    best_assignment: pd.DataFrame | None = None
    best_record: dict[str, Any] | None = None

    for attempt in attempts:
        derived_seed = seed * 1_000_003 + int(attempt)
        assignment, objective = assign_clusters_once(
            cluster_table,
            proportions,
            weights,
            overflow_penalty,
            derived_seed,
        )
        metrics = assignment_metrics(assignment, proportions, global_prevalence)
        feasible = metrics_are_feasible(metrics, validation)
        record = {
            "attempt": int(attempt),
            "derived_seed": int(derived_seed),
            "objective": float(objective),
            "feasible": int(feasible),
            **metrics,
        }

        if best_record is None:
            best_assignment = assignment
            best_record = record
        else:
            current_key = (-record["feasible"], record["objective"])
            best_key = (-best_record["feasible"], best_record["objective"])
            if current_key < best_key:
                best_assignment = assignment
                best_record = record

    if best_assignment is None or best_record is None:
        raise RuntimeError("No group assignment was generated.")

    # An audit-selected attempt may fail if the final validation tolerances
    # are stricter than those used during the structural audit. In that
    # case, search the complete configured attempt range.
    if not best_record["feasible"] and preferred_attempt is not None:
        return select_group_assignment(
            cluster_table=cluster_table,
            proportions=proportions,
            weights=weights,
            overflow_penalty=overflow_penalty,
            seed=seed,
            attempts_per_seed=attempts_per_seed,
            validation=validation,
            preferred_attempt=None,
        )

    if not best_record["feasible"]:
        raise RuntimeError(
            f"No feasible assignment was found for seed {seed} after "
            f"{len(attempts)} attempt(s)."
        )
    return best_assignment, best_record


def load_membership(
    config: dict[str, Any],
    task_id: str,
    homology_config_id: str,
    task_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    root = Path(config["input"]["homology_root"])
    path = root / task_id / homology_config_id / "cluster_membership.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cluster membership not found: {path}")

    membership = pd.read_csv(path, keep_default_na=False)
    required = [
        "task_id",
        "homology_config_id",
        "sequence_id",
        "cluster_id",
        "amp_label",
        "minimum_sequence_identity",
    ]
    missing = [column for column in required if column not in membership.columns]
    if missing:
        raise ValueError(f"Missing membership columns in {path}: {', '.join(missing)}")

    if set(membership["task_id"].astype(str)) != {task_id}:
        raise ValueError(f"Unexpected task IDs in {path}")
    if set(membership["homology_config_id"].astype(str)) != {homology_config_id}:
        raise ValueError(f"Unexpected homology configuration IDs in {path}")
    if membership["sequence_id"].duplicated().any():
        raise ValueError(f"Duplicated sequence IDs in {path}")

    dataset_config = config["dataset"]
    sequence_id_column = dataset_config["sequence_id_column"]
    label_column = dataset_config["label_column"]

    expected = task_dataframe[[sequence_id_column, label_column]].rename(
        columns={sequence_id_column: "sequence_id", label_column: "expected_label"}
    )
    membership = membership.merge(expected, on="sequence_id", how="left", validate="one_to_one")

    if membership["expected_label"].isna().any():
        raise ValueError(f"Membership contains IDs absent from the task dataset: {path}")
    membership["amp_label"] = pd.to_numeric(membership["amp_label"], errors="raise").astype(int)
    if (membership["amp_label"] != membership["expected_label"]).any():
        raise ValueError(f"Label mismatch between membership and task dataset: {path}")

    expected_ids = set(task_dataframe[sequence_id_column].astype(str))
    observed_ids = set(membership["sequence_id"].astype(str))
    if expected_ids != observed_ids:
        raise ValueError(f"Membership IDs do not match the task dataset: {path}")

    threshold_values = pd.to_numeric(
        membership["minimum_sequence_identity"], errors="raise"
    ).unique()
    if len(threshold_values) != 1:
        raise ValueError(f"Multiple identity thresholds were found in {path}")

    return membership.drop(columns="expected_label"), float(threshold_values[0])


def validate_sequence_assignment(
    assignment: pd.DataFrame,
    expected_ids: set[str],
    proportions: dict[str, float],
    validation: dict[str, Any],
    require_group_integrity: bool,
) -> dict[str, Any]:
    partitions = {"train", "validation", "test"}
    observed_partitions = set(assignment["partition"].astype(str))

    duplicate_count = int(assignment["sequence_id"].duplicated().sum())
    observed_ids = set(assignment["sequence_id"].astype(str))
    missing_ids = expected_ids - observed_ids
    extra_ids = observed_ids - expected_ids

    group_violations = 0
    if require_group_integrity:
        group_violations = int(
            (assignment.groupby("group_id")["partition"].nunique() > 1).sum()
        )

    counts = assignment.groupby("partition").agg(
        sequences=("sequence_id", "size"),
        positives=("amp_label", "sum"),
    )
    counts["negatives"] = counts["sequences"] - counts["positives"]
    counts["prevalence"] = counts["positives"] / counts["sequences"]

    global_prevalence = float(assignment["amp_label"].mean())
    max_size_deviation = 0.0
    max_prevalence_deviation = 0.0

    result: dict[str, Any] = {}
    for partition, fraction in proportions.items():
        if partition not in counts.index:
            total = positives = negatives = 0
            prevalence = math.nan
        else:
            row = counts.loc[partition]
            total = int(row["sequences"])
            positives = int(row["positives"])
            negatives = int(row["negatives"])
            prevalence = float(row["prevalence"])

        target = len(assignment) * fraction
        size_deviation = abs(total - target) / max(target, 1.0)
        prevalence_deviation = (
            abs(prevalence - global_prevalence) if total else math.inf
        )

        result[f"{partition}_sequences"] = total
        result[f"{partition}_positives"] = positives
        result[f"{partition}_negatives"] = negatives
        result[f"{partition}_prevalence"] = prevalence
        result[f"{partition}_size_deviation"] = size_deviation
        result[f"{partition}_prevalence_deviation"] = prevalence_deviation

        max_size_deviation = max(max_size_deviation, size_deviation)
        max_prevalence_deviation = max(max_prevalence_deviation, prevalence_deviation)

    result.update(
        {
            "duplicate_sequence_count": duplicate_count,
            "missing_sequence_count": len(missing_ids),
            "unexpected_sequence_count": len(extra_ids),
            "missing_partition_count": len(partitions - observed_partitions),
            "group_cross_partition_violations": group_violations,
            "maximum_split_size_deviation": max_size_deviation,
            "maximum_prevalence_deviation": max_prevalence_deviation,
        }
    )

    basic_valid = all(
        [
            duplicate_count == 0,
            not missing_ids,
            not extra_ids,
            observed_partitions == partitions,
            group_violations == 0,
            all(result[f"{partition}_positives"] > 0 for partition in partitions),
            all(result[f"{partition}_negatives"] > 0 for partition in partitions),
            max_size_deviation <= float(validation["maximum_split_size_deviation"]),
            max_prevalence_deviation <= float(validation["maximum_prevalence_deviation"]),
        ]
    )
    result["validation_passed"] = int(basic_valid)
    return result


def expand_group_assignment(
    membership: pd.DataFrame,
    cluster_assignment: pd.DataFrame,
) -> pd.DataFrame:
    mapping = cluster_assignment[["cluster_id", "partition"]]
    expanded = membership[["sequence_id", "amp_label", "cluster_id"]].merge(
        mapping,
        on="cluster_id",
        how="left",
        validate="many_to_one",
    )
    return expanded.rename(columns={"cluster_id": "group_id"})


def preferred_attempt_for_seed(
    audit_runs: pd.DataFrame | None,
    task_id: str,
    config_id: str,
    seed: int,
    enabled: bool,
) -> int | None:
    if not enabled or audit_runs is None:
        return None
    required = {"task_id", "homology_config_id", "seed", "attempt", "feasible"}
    if not required.issubset(audit_runs.columns):
        raise ValueError("The homology feasibility table is missing required columns.")

    subset = audit_runs[
        (audit_runs["task_id"].astype(str) == task_id)
        & (audit_runs["homology_config_id"].astype(str) == config_id)
        & (pd.to_numeric(audit_runs["seed"], errors="coerce") == seed)
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    if int(row["feasible"]) != 1:
        return None
    return int(row["attempt"])


def check_audit_eligibility(
    audit_summary: pd.DataFrame | None,
    task_id: str,
    config_id: str,
    require_eligible: bool,
) -> None:
    if not require_eligible:
        return
    if audit_summary is None:
        raise ValueError("Audit eligibility is required, but no audit summary was provided.")
    required = {"task_id", "homology_config_id", "selection_status"}
    if not required.issubset(audit_summary.columns):
        raise ValueError("The homology audit summary is missing required columns.")

    subset = audit_summary[
        (audit_summary["task_id"].astype(str) == task_id)
        & (audit_summary["homology_config_id"].astype(str) == config_id)
    ]
    if len(subset) != 1:
        raise ValueError(f"Expected one audit row for {task_id}/{config_id}.")
    status = str(subset.iloc[0]["selection_status"])
    if status != "eligible":
        raise ValueError(f"Configuration {task_id}/{config_id} is not eligible: {status}")


def write_assignment(dataframe: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv.gz":
        dataframe.to_csv(path, index=False, compression="gzip")
    else:
        dataframe.to_csv(path, index=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 30 reproducible random-stratified and MMseqs2 "
            "group-aware partition sets for each AMP task."
        )
    )
    parser.add_argument("--config", required=True, help="Path to partitioning YAML.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    config_path = Path(args.config)

    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_yaml(config_path)
        validate_config(config)

        dataframe, task_dataset_path = load_task_dataset(config)
        audit_summary = load_audit_summary(config)
        audit_runs = load_audit_runs(config)

        dataset_config = config["dataset"]
        partition_config = config["partitioning"]
        validation_config = config["validation"]
        output_config = config["output"]

        task_column = dataset_config["task_column"]
        sequence_id_column = dataset_config["sequence_id_column"]
        label_column = dataset_config["label_column"]

        proportions = {
            name: float(value)
            for name, value in partition_config["proportions"].items()
        }
        number_of_seeds = int(partition_config["number_of_seeds"])
        first_seed = int(partition_config.get("first_seed", 0))
        attempts_per_seed = int(partition_config["attempts_per_seed"])
        weights = {
            name: float(value)
            for name, value in partition_config.get("objective_weights", {}).items()
        }
        overflow_penalty = float(partition_config.get("overflow_penalty", 2.0))
        reuse_audit_attempts = bool(partition_config.get("reuse_audit_attempts", True))
        require_audit_eligibility = bool(
            partition_config.get("require_audit_eligibility", True)
        )

        output_root = Path(output_config["partition_root"])
        run_index_path = Path(output_config["partition_index"])
        validation_path = Path(output_config["validation_summary"])
        manifest_path = Path(output_config["manifest"])
        output_format = str(output_config.get("assignment_format", "csv.gz"))

        principal_outputs = [run_index_path, validation_path, manifest_path]
        if output_root.exists():
            if args.overwrite:
                shutil.rmtree(output_root)
            else:
                raise FileExistsError(
                    f"Partition root already exists: {output_root}. Use --overwrite."
                )
        if not args.overwrite:
            existing = [path for path in principal_outputs if path.exists()]
            if existing:
                raise FileExistsError(
                    "Output files already exist: " + ", ".join(map(str, existing))
                )
        else:
            for path in principal_outputs:
                if path.exists():
                    path.unlink()

        output_root.mkdir(parents=True, exist_ok=False)

        random_enabled = bool(partition_config.get("random_stratified", {}).get("enabled", True))
        random_regime_id = str(
            partition_config.get("random_stratified", {}).get("regime_id", "RANDOM")
        )
        homology_configs = [
            str(value) for value in partition_config.get("homology_configurations", [])
        ]

        run_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        hashes_by_regime: dict[tuple[str, str], list[str]] = {}

        for task_id in sorted(dataframe[task_column].unique()):
            task_dataframe = dataframe[dataframe[task_column] == task_id].copy()
            task_dataframe = task_dataframe.sort_values(sequence_id_column).reset_index(drop=True)
            expected_ids = set(task_dataframe[sequence_id_column].astype(str))

            regimes: list[tuple[str, str | None]] = []
            if random_enabled:
                regimes.append((random_regime_id, None))
            regimes.extend((config_id, config_id) for config_id in homology_configs)

            for regime_id, homology_config_id in regimes:
                membership = None
                threshold = None
                cluster_table = None

                if homology_config_id is not None:
                    check_audit_eligibility(
                        audit_summary,
                        task_id,
                        homology_config_id,
                        require_audit_eligibility,
                    )
                    membership, threshold = load_membership(
                        config,
                        task_id,
                        homology_config_id,
                        task_dataframe,
                    )
                    cluster_table = build_cluster_table(membership)

                hashes_by_regime[(task_id, regime_id)] = []

                for seed in range(first_seed, first_seed + number_of_seeds):
                    if homology_config_id is None:
                        assignment = build_random_stratified_assignment(
                            task_dataframe,
                            sequence_id_column,
                            label_column,
                            proportions,
                            seed,
                        )
                        selection_record = {
                            "attempt": None,
                            "derived_seed": seed,
                            "objective": None,
                            "feasible": 1,
                        }
                        regime_name = "random_stratified"
                        similarity_threshold = None
                    else:
                        preferred_attempt = preferred_attempt_for_seed(
                            audit_runs,
                            task_id,
                            homology_config_id,
                            seed,
                            reuse_audit_attempts,
                        )
                        cluster_assignment, selection_record = select_group_assignment(
                            cluster_table,
                            proportions,
                            weights,
                            overflow_penalty,
                            seed,
                            attempts_per_seed,
                            validation_config,
                            preferred_attempt,
                        )
                        assignment = expand_group_assignment(membership, cluster_assignment)
                        regime_name = f"homology_aware_{homology_config_id}"
                        similarity_threshold = threshold

                    assignment = assignment.rename(
                        columns={sequence_id_column: "sequence_id", label_column: "amp_label"}
                    )
                    assignment["task_id"] = task_id
                    assignment["similarity_regime"] = regime_name
                    assignment["similarity_threshold"] = similarity_threshold
                    assignment["seed"] = seed

                    assignment = assignment[
                        [
                            "task_id",
                            "sequence_id",
                            "amp_label",
                            "partition",
                            "similarity_regime",
                            "similarity_threshold",
                            "group_id",
                            "seed",
                        ]
                    ].sort_values("sequence_id").reset_index(drop=True)

                    validation_result = validate_sequence_assignment(
                        assignment,
                        expected_ids,
                        proportions,
                        validation_config,
                        require_group_integrity=True,
                    )
                    if not validation_result["validation_passed"]:
                        raise RuntimeError(
                            f"Validation failed for {task_id}/{regime_id}/seed {seed}."
                        )

                    digest = assignment_hash(assignment)
                    split_id = stable_split_id(task_id, regime_id, seed, digest)
                    assignment["split_id"] = split_id

                    # Deterministic regeneration check.
                    if homology_config_id is None:
                        repeated = build_random_stratified_assignment(
                            task_dataframe,
                            sequence_id_column,
                            label_column,
                            proportions,
                            seed,
                        )
                    else:
                        repeated_cluster_assignment, _ = select_group_assignment(
                            cluster_table,
                            proportions,
                            weights,
                            overflow_penalty,
                            seed,
                            attempts_per_seed,
                            validation_config,
                            int(selection_record["attempt"]),
                        )
                        repeated = expand_group_assignment(membership, repeated_cluster_assignment)
                    repeated = repeated.rename(
                        columns={sequence_id_column: "sequence_id", label_column: "amp_label"}
                    )
                    repeated = repeated[["sequence_id", "partition"]].sort_values("sequence_id")
                    reproducible = int(
                        digest
                        == assignment_hash(
                            repeated.assign(
                                amp_label=0,
                                group_id="",
                            )[["sequence_id", "partition"]]
                        )
                    )
                    if reproducible != 1:
                        raise RuntimeError(
                            f"Reproducibility check failed for {task_id}/{regime_id}/seed {seed}."
                        )

                    seed_directory = output_root / task_id / regime_id / f"seed_{seed:03d}"
                    extension = "csv.gz" if output_format == "csv.gz" else "csv"
                    assignment_path = seed_directory / f"split_assignment.{extension}"
                    write_assignment(assignment, assignment_path, output_format)

                    validation_record = {
                        "task_id": task_id,
                        "regime_id": regime_id,
                        "similarity_regime": regime_name,
                        "similarity_threshold": similarity_threshold,
                        "seed": seed,
                        "split_id": split_id,
                        "assignment_sha256": digest,
                        "selected_attempt": selection_record.get("attempt"),
                        "derived_seed": selection_record.get("derived_seed"),
                        "assignment_objective": selection_record.get("objective"),
                        "reproducibility_check_passed": reproducible,
                        "cross_split_similarity_validation_mode": (
                            "singleton_groups"
                            if homology_config_id is None
                            else "mmseqs_connected_component_integrity"
                        ),
                        "cross_split_similarity_violation_count": 0,
                        "maximum_cross_split_identity": None,
                        "maximum_cross_split_identity_status": "not_directly_measured",
                        **validation_result,
                    }
                    validation_rows.append(validation_record)

                    split_manifest = {
                        "schema_version": "1.0",
                        "created_at_utc": utc_now(),
                        "task_id": task_id,
                        "regime_id": regime_id,
                        "similarity_regime": regime_name,
                        "similarity_threshold": similarity_threshold,
                        "seed": seed,
                        "split_id": split_id,
                        "assignment_sha256": digest,
                        "source_task_dataset": {
                            "path": str(task_dataset_path),
                            "sha256": sha256_file(task_dataset_path),
                        },
                        "selection": selection_record,
                        "validation": validation_record,
                        "assignment_file": {
                            "path": str(assignment_path),
                            "sha256": sha256_file(assignment_path),
                            "size_bytes": assignment_path.stat().st_size,
                        },
                    }
                    split_manifest_path = seed_directory / "manifest.json"
                    write_json(split_manifest, split_manifest_path)

                    run_rows.append(
                        {
                            "task_id": task_id,
                            "regime_id": regime_id,
                            "similarity_regime": regime_name,
                            "similarity_threshold": similarity_threshold,
                            "seed": seed,
                            "split_id": split_id,
                            "assignment_path": str(assignment_path),
                            "manifest_path": str(split_manifest_path),
                            "assignment_sha256": digest,
                            "file_sha256": sha256_file(assignment_path),
                            "selected_attempt": selection_record.get("attempt"),
                            "validation_passed": validation_result["validation_passed"],
                        }
                    )
                    hashes_by_regime[(task_id, regime_id)].append(digest)

                    print(
                        f"{task_id} / {regime_id} / seed {seed}: "
                        f"train={validation_result['train_sequences']} "
                        f"validation={validation_result['validation_sequences']} "
                        f"test={validation_result['test_sequences']}"
                    )

        require_unique = bool(validation_config.get("require_unique_partition_sets", True))
        uniqueness_summary: dict[str, Any] = {}
        for (task_id, regime_id), digests in hashes_by_regime.items():
            unique_count = len(set(digests))
            uniqueness_summary[f"{task_id}/{regime_id}"] = {
                "requested_partition_sets": len(digests),
                "unique_partition_sets": unique_count,
            }
            if require_unique and unique_count != len(digests):
                raise RuntimeError(
                    f"Duplicate partition sets were generated for {task_id}/{regime_id}: "
                    f"{unique_count}/{len(digests)} unique."
                )

        run_index = pd.DataFrame(run_rows).sort_values(["task_id", "regime_id", "seed"])
        validation_summary = pd.DataFrame(validation_rows).sort_values(
            ["task_id", "regime_id", "seed"]
        )

        run_index_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        run_index.to_csv(run_index_path, index=False)
        validation_summary.to_csv(validation_path, index=False)

        manifest = {
            "schema_version": "1.0",
            "created_at_utc": utc_now(),
            "purpose": (
                "Generation of reproducible random-stratified and MMseqs2 "
                "group-constrained AMP benchmark partitions."
            ),
            "number_of_seeds": number_of_seeds,
            "first_seed": first_seed,
            "proportions": proportions,
            "configuration": config,
            "task_dataset": {
                "path": str(task_dataset_path),
                "sha256": sha256_file(task_dataset_path),
            },
            "partition_sets": len(run_index),
            "uniqueness": uniqueness_summary,
            "cross_split_similarity_note": (
                "Homology-aware partitions validate that no MMseqs2 connected component "
                "crosses partitions. Exact numeric maximum cross-split identity is not "
                "available from cluster-membership tables alone and is therefore not "
                "reported by this script."
            ),
            "outputs": {
                "partition_root": str(output_root),
                "partition_index": {
                    "path": str(run_index_path),
                    "sha256": sha256_file(run_index_path),
                },
                "validation_summary": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
            },
        }
        write_json(manifest, manifest_path)

        print("\nPartition generation completed")
        print(f"Partition sets: {len(run_index)}")
        print(f"Partition index: {run_index_path}")
        print(f"Validation summary: {validation_path}")
        print(f"Manifest: {manifest_path}")
        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        pd.errors.ParserError,
        yaml.YAMLError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
