#!/usr/bin/env python3

"""Run task-specific MMseqs2 clustering for homology-aware partitioning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def decode_separator(value: str) -> str:
    """Convert escaped separators such as '\\t' into actual characters."""
    decoded = bytes(value, "utf-8").decode("unicode_escape")

    if len(decoded) != 1:
        raise ValueError(
            "The separator must resolve to exactly one character."
        )

    return decoded


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def validate_safe_id(
    value: str,
    field_name: str,
) -> None:
    """Validate identifiers used as directory names."""
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} '{value}' contains unsupported characters. "
            "Use only letters, numbers, '.', '_' and '-'."
        )


def validate_config(config: dict[str, Any]) -> None:
    """Validate the required configuration fields."""
    required_sections = (
        "input",
        "dataset",
        "mmseqs",
        "configurations",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    dataset = config["dataset"]

    required_dataset_fields = (
        "task_column",
        "sequence_id_column",
        "sequence_column",
        "label_column",
        "length_column",
        "alphabet",
    )

    for field in required_dataset_fields:
        if field not in dataset:
            raise ValueError(
                f"Missing dataset configuration field: {field}"
            )

    mmseqs = config["mmseqs"]

    required_mmseqs_fields = (
        "binary",
        "threads",
        "cluster_mode",
        "alignment_mode",
        "coverage_mode",
        "minimum_coverage",
    )

    for field in required_mmseqs_fields:
        if field not in mmseqs:
            raise ValueError(
                f"Missing MMseqs2 configuration field: {field}"
            )

    if int(mmseqs["threads"]) < 1:
        raise ValueError(
            "mmseqs.threads must be greater than zero."
        )

    minimum_coverage = float(
        mmseqs["minimum_coverage"]
    )

    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError(
            "mmseqs.minimum_coverage must be between 0 and 1."
        )

    configurations = config["configurations"]

    if not isinstance(configurations, dict) or not configurations:
        raise ValueError(
            "configurations must contain at least one mapping."
        )

    enabled_configurations = 0

    for config_id, definition in configurations.items():
        config_id = str(config_id)
        validate_safe_id(
            config_id,
            "configuration ID",
        )

        if not isinstance(definition, dict):
            raise ValueError(
                f"Configuration '{config_id}' must be a YAML mapping."
            )

        if not definition.get("enabled", True):
            continue

        enabled_configurations += 1

        identity = float(
            definition.get(
                "minimum_sequence_identity",
                -1,
            )
        )

        if not 0.0 <= identity <= 1.0:
            raise ValueError(
                f"Configuration '{config_id}' has an invalid "
                "minimum_sequence_identity."
            )

    if enabled_configurations == 0:
        raise ValueError(
            "No enabled clustering configurations were found."
        )


def resolve_mmseqs_binary(
    configured: str,
    override: str | None,
) -> str:
    """Resolve the MMseqs2 executable."""
    candidate = override or configured
    resolved = shutil.which(candidate)

    if resolved is None:
        raise FileNotFoundError(
            f"MMseqs2 executable '{candidate}' was not found in PATH."
        )

    return resolved


def get_mmseqs_version(binary: str) -> str:
    """Read the MMseqs2 version string."""
    completed = subprocess.run(
        [binary, "version"],
        check=True,
        capture_output=True,
        text=True,
    )

    version = (
        completed.stdout
        or completed.stderr
    ).strip()

    if not version:
        raise RuntimeError(
            "MMseqs2 returned an empty version string."
        )

    return version


def load_dataset(
    input_path: Path,
    config: dict[str, Any],
    task_filter: set[str] | None,
) -> pd.DataFrame:
    """Load and validate the task dataset."""
    input_config = config["input"]
    dataset = config["dataset"]

    separator = decode_separator(
        str(input_config.get("separator", ","))
    )
    encoding = str(
        input_config.get("encoding", "utf-8")
    )

    task_column = dataset["task_column"]
    sequence_id_column = dataset[
        "sequence_id_column"
    ]
    sequence_column = dataset["sequence_column"]
    label_column = dataset["label_column"]
    length_column = dataset["length_column"]

    dataframe = pd.read_csv(
        input_path,
        sep=separator,
        encoding=encoding,
        keep_default_na=False,
        dtype={
            task_column: str,
            sequence_id_column: str,
            sequence_column: str,
        },
    )

    required_columns = [
        task_column,
        sequence_id_column,
        sequence_column,
        label_column,
        length_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns were not found: "
            + ", ".join(missing_columns)
        )

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
    dataframe[sequence_column] = (
        dataframe[sequence_column]
        .astype(str)
        .str.strip()
    )

    configured_tasks = dataset.get(
        "included_tasks"
    )

    if configured_tasks:
        configured_task_set = {
            str(task)
            for task in configured_tasks
        }

        dataframe = dataframe[
            dataframe[task_column].isin(
                configured_task_set
            )
        ].copy()

    if task_filter:
        available_before_filter = set(
            dataframe[task_column].unique()
        )

        missing_requested_tasks = (
            task_filter - available_before_filter
        )

        if missing_requested_tasks:
            raise ValueError(
                "Requested tasks were not found: "
                + ", ".join(
                    sorted(missing_requested_tasks)
                )
            )

        dataframe = dataframe[
            dataframe[task_column].isin(
                task_filter
            )
        ].copy()

    if dataframe.empty:
        raise ValueError(
            "No rows remain after task filtering."
        )

    for task_id in dataframe[task_column].unique():
        validate_safe_id(
            str(task_id),
            "task ID",
        )

    if (
        dataframe[sequence_id_column] == ""
    ).any():
        raise ValueError(
            "Empty sequence identifiers were found."
        )

    if (
        dataframe[sequence_column] == ""
    ).any():
        raise ValueError(
            "Empty sequences were found."
        )

    duplicate_mask = dataframe.duplicated(
        subset=[
            task_column,
            sequence_id_column,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        examples = dataframe.loc[
            duplicate_mask,
            [
                task_column,
                sequence_id_column,
            ],
        ].head(5)

        formatted_examples = []

        for _, row in examples.iterrows():
            formatted_examples.append(
                f"{row[task_column]}:"
                f"{row[sequence_id_column]}"
            )

        raise ValueError(
            "Duplicated task/sequence combinations were found: "
            + "; ".join(formatted_examples)
        )

    id_sequence_counts = dataframe.groupby(
        sequence_id_column
    )[sequence_column].nunique()

    inconsistent_ids = id_sequence_counts[
        id_sequence_counts > 1
    ]

    if not inconsistent_ids.empty:
        raise ValueError(
            "Some sequence IDs map to multiple sequences. "
            "Examples: "
            + ", ".join(
                inconsistent_ids.index.astype(str)[:5]
            )
        )

    dataframe[label_column] = pd.to_numeric(
        dataframe[label_column],
        errors="raise",
    ).astype(int)

    unexpected_labels = sorted(
        set(dataframe[label_column].unique())
        - {0, 1}
    )

    if unexpected_labels:
        raise ValueError(
            "Only binary labels 0 and 1 are supported. Found: "
            + ", ".join(
                map(str, unexpected_labels)
            )
        )

    dataframe[length_column] = pd.to_numeric(
        dataframe[length_column],
        errors="raise",
    ).astype(int)

    if dataset.get(
        "verify_reported_length",
        True,
    ):
        actual_lengths = dataframe[
            sequence_column
        ].str.len()

        mismatch_mask = (
            actual_lengths
            != dataframe[length_column]
        )

        if mismatch_mask.any():
            examples = dataframe.loc[
                mismatch_mask,
                sequence_id_column,
            ].head(5).tolist()

            raise ValueError(
                "Reported lengths do not match sequence lengths. "
                "Examples: "
                + ", ".join(examples)
            )

    alphabet = set(
        str(dataset["alphabet"])
    )

    invalid_examples: list[str] = []

    unique_sequences = dataframe[
        [
            sequence_id_column,
            sequence_column,
        ]
    ].drop_duplicates()

    for sequence_id, sequence in unique_sequences.itertuples(
        index=False,
        name=None,
    ):
        invalid_characters = sorted(
            set(sequence) - alphabet
        )

        if invalid_characters:
            invalid_examples.append(
                f"{sequence_id}:"
                f"{''.join(invalid_characters)}"
            )

            if len(invalid_examples) == 5:
                break

    if invalid_examples:
        raise ValueError(
            "Sequences contain characters outside the configured "
            "alphabet: "
            + "; ".join(invalid_examples)
        )

    class_counts = dataframe.groupby(
        task_column
    )[label_column].nunique()

    incomplete_tasks = class_counts[
        class_counts < 2
    ]

    if not incomplete_tasks.empty:
        raise ValueError(
            "The following tasks do not contain both classes: "
            + ", ".join(
                incomplete_tasks.index.astype(str)
            )
        )

    return dataframe


def write_fasta(
    task_dataframe: pd.DataFrame,
    sequence_id_column: str,
    sequence_column: str,
    output_path: Path,
) -> None:
    """Write a deterministic FASTA sorted by sequence identifier."""
    records = task_dataframe[
        [
            sequence_id_column,
            sequence_column,
        ]
    ].sort_values(sequence_id_column)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for sequence_id, sequence in records.itertuples(
            index=False,
            name=None,
        ):
            if (
                any(
                    character.isspace()
                    for character in sequence_id
                )
                or ">" in sequence_id
            ):
                raise ValueError(
                    f"Sequence ID is not FASTA-safe: {sequence_id}"
                )

            handle.write(
                f">{sequence_id}\n"
            )

            for start in range(
                0,
                len(sequence),
                80,
            ):
                handle.write(
                    sequence[start:start + 80]
                    + "\n"
                )


def run_command(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run a command and append its output to a log file."""
    started = time.monotonic()
    command_text = shlex.join(command)

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            f"\n[{utc_now()}] COMMAND\n"
            f"{command_text}\n\n"
        )
        log_handle.flush()

        completed = subprocess.run(
            command,
            check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )

        elapsed_seconds = (
            time.monotonic() - started
        )

        log_handle.write(
            f"\n[{utc_now()}] "
            f"RETURN_CODE={completed.returncode} "
            f"ELAPSED_SECONDS={elapsed_seconds:.6f}\n"
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {command_text}. "
            f"See {log_path}."
        )

    return {
        "command": command_text,
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
    }


def build_cluster_command(
    binary: str,
    sequence_db: Path,
    cluster_db: Path,
    temporary_directory: Path,
    mmseqs_config: dict[str, Any],
    identity: float,
    config_definition: dict[str, Any],
) -> list[str]:
    """Build the MMseqs2 cluster command."""
    command = [
        binary,
        "cluster",
        str(sequence_db),
        str(cluster_db),
        str(temporary_directory),
        "--min-seq-id",
        str(identity),
        "-c",
        str(
            float(
                mmseqs_config[
                    "minimum_coverage"
                ]
            )
        ),
        "--cov-mode",
        str(
            int(
                mmseqs_config[
                    "coverage_mode"
                ]
            )
        ),
        "--cluster-mode",
        str(
            int(
                mmseqs_config[
                    "cluster_mode"
                ]
            )
        ),
        "--alignment-mode",
        str(
            int(
                mmseqs_config[
                    "alignment_mode"
                ]
            )
        ),
        "--threads",
        str(
            int(
                mmseqs_config["threads"]
            )
        ),
    ]

    optional_parameters = {
        "e_value": "-e",
        "sensitivity": "-s",
        "max_sequences": "--max-seqs",
        "split_memory_limit": (
            "--split-memory-limit"
        ),
    }

    for parameter_name, flag in (
        optional_parameters.items()
    ):
        value = config_definition.get(
            parameter_name,
            mmseqs_config.get(
                parameter_name
            ),
        )

        if value is not None:
            command.extend(
                [
                    flag,
                    str(value),
                ]
            )

    extra_arguments = list(
        mmseqs_config.get(
            "extra_args",
            [],
        )
    )

    extra_arguments.extend(
        config_definition.get(
            "extra_args",
            [],
        )
    )

    command.extend(
        str(argument)
        for argument in extra_arguments
    )

    return command


def parse_cluster_tsv(
    tsv_path: Path,
    expected_sequence_ids: set[str],
) -> dict[str, list[str]]:
    """Parse MMseqs2 representative/member TSV output."""
    representative_by_member: dict[
        str,
        str,
    ] = {}

    with tsv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        for line_number, row in enumerate(
            reader,
            start=1,
        ):
            if not row:
                continue

            if len(row) != 2:
                raise ValueError(
                    "Unexpected MMseqs2 TSV row at line "
                    f"{line_number}: {row}"
                )

            representative = row[0].strip()
            member = row[1].strip()

            if representative not in expected_sequence_ids:
                raise ValueError(
                    "Unknown representative sequence ID in "
                    f"MMseqs2 output: {representative}"
                )

            if member not in expected_sequence_ids:
                raise ValueError(
                    "Unknown member sequence ID in MMseqs2 output: "
                    f"{member}"
                )

            previous_representative = (
                representative_by_member.get(
                    member
                )
            )

            if (
                previous_representative is not None
                and previous_representative
                != representative
            ):
                raise ValueError(
                    f"Sequence '{member}' appears in multiple "
                    f"clusters: '{previous_representative}' and "
                    f"'{representative}'."
                )

            representative_by_member[
                member
            ] = representative

    observed_members = set(
        representative_by_member
    )

    missing_ids = sorted(
        expected_sequence_ids
        - observed_members
    )

    unexpected_ids = sorted(
        observed_members
        - expected_sequence_ids
    )

    if missing_ids or unexpected_ids:
        messages = []

        if missing_ids:
            messages.append(
                "missing IDs: "
                + ", ".join(
                    missing_ids[:10]
                )
            )

        if unexpected_ids:
            messages.append(
                "unexpected IDs: "
                + ", ".join(
                    unexpected_ids[:10]
                )
            )

        raise ValueError(
            "Invalid MMseqs2 cluster membership; "
            + "; ".join(messages)
        )

    clusters: dict[
        str,
        list[str],
    ] = {}

    for member, representative in (
        representative_by_member.items()
    ):
        clusters.setdefault(
            representative,
            [],
        ).append(member)

    for members in clusters.values():
        members.sort()

    return clusters


def stable_cluster_id(
    task_id: str,
    config_id: str,
    members: list[str],
    prefix_length: int,
) -> tuple[str, str]:
    """Create a stable cluster ID from sorted cluster members."""
    hash_material = "\n".join(
        [
            task_id,
            config_id,
            *members,
        ]
    ).encode("utf-8")

    full_hash = hashlib.sha256(
        hash_material
    ).hexdigest()

    cluster_id = (
        f"{config_id}_"
        f"{full_hash[:prefix_length]}"
    )

    return cluster_id, full_hash


def build_cluster_tables(
    task_dataframe: pd.DataFrame,
    clusters: dict[str, list[str]],
    task_id: str,
    config_id: str,
    identity: float,
    mmseqs_config: dict[str, Any],
    dataset_config: dict[str, Any],
    hash_prefix_length: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Build membership, cluster-level, and global summaries."""
    sequence_id_column = dataset_config[
        "sequence_id_column"
    ]
    label_column = dataset_config[
        "label_column"
    ]
    length_column = dataset_config[
        "length_column"
    ]

    indexed = task_dataframe.set_index(
        sequence_id_column,
        verify_integrity=True,
    )

    membership_rows: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    for representative, members in sorted(
        clusters.items()
    ):
        cluster_id, cluster_hash = (
            stable_cluster_id(
                task_id=task_id,
                config_id=config_id,
                members=members,
                prefix_length=(
                    hash_prefix_length
                ),
            )
        )

        member_data = indexed.loc[
            members
        ]

        labels = member_data[
            label_column
        ].astype(int)

        lengths = member_data[
            length_column
        ].astype(int)

        positive_count = int(
            (labels == 1).sum()
        )

        negative_count = int(
            (labels == 0).sum()
        )

        cluster_size = len(members)

        mixed_label_cluster = (
            positive_count > 0
            and negative_count > 0
        )

        summary_rows.append(
            {
                "task_id": task_id,
                "homology_config_id": (
                    config_id
                ),
                "minimum_sequence_identity": (
                    identity
                ),
                "minimum_coverage": float(
                    mmseqs_config[
                        "minimum_coverage"
                    ]
                ),
                "cluster_id": cluster_id,
                "cluster_hash_sha256": (
                    cluster_hash
                ),
                "representative_sequence_id": (
                    representative
                ),
                "cluster_size": cluster_size,
                "positive_count": (
                    positive_count
                ),
                "negative_count": (
                    negative_count
                ),
                "positive_fraction": (
                    positive_count
                    / cluster_size
                ),
                "mixed_label_cluster": int(
                    mixed_label_cluster
                ),
                "minimum_length": int(
                    lengths.min()
                ),
                "maximum_length": int(
                    lengths.max()
                ),
                "median_length": float(
                    statistics.median(
                        lengths.tolist()
                    )
                ),
            }
        )

        for member in members:
            membership_rows.append(
                {
                    "task_id": task_id,
                    "homology_config_id": (
                        config_id
                    ),
                    "minimum_sequence_identity": (
                        identity
                    ),
                    "minimum_coverage": float(
                        mmseqs_config[
                            "minimum_coverage"
                        ]
                    ),
                    "coverage_mode": int(
                        mmseqs_config[
                            "coverage_mode"
                        ]
                    ),
                    "cluster_mode": int(
                        mmseqs_config[
                            "cluster_mode"
                        ]
                    ),
                    "sequence_id": member,
                    "representative_sequence_id": (
                        representative
                    ),
                    "cluster_id": cluster_id,
                    "amp_label": int(
                        indexed.at[
                            member,
                            label_column,
                        ]
                    ),
                    "sequence_length": int(
                        indexed.at[
                            member,
                            length_column,
                        ]
                    ),
                    "cluster_size": (
                        cluster_size
                    ),
                    "cluster_positive_count": (
                        positive_count
                    ),
                    "cluster_negative_count": (
                        negative_count
                    ),
                    "mixed_label_cluster": int(
                        mixed_label_cluster
                    ),
                }
            )

    membership = pd.DataFrame(
        membership_rows
    ).sort_values(
        [
            "cluster_id",
            "sequence_id",
        ]
    ).reset_index(drop=True)

    cluster_summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "cluster_size",
            "cluster_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    cluster_sizes = cluster_summary[
        "cluster_size"
    ].astype(int)

    mixed_mask = (
        cluster_summary[
            "mixed_label_cluster"
        ]
        == 1
    )

    global_summary = {
        "number_of_sequences": int(
            len(membership)
        ),
        "number_of_clusters": int(
            len(cluster_summary)
        ),
        "number_of_singletons": int(
            (cluster_sizes == 1).sum()
        ),
        "singleton_fraction": float(
            (cluster_sizes == 1).mean()
        ),
        "largest_cluster_size": int(
            cluster_sizes.max()
        ),
        "mean_cluster_size": float(
            cluster_sizes.mean()
        ),
        "median_cluster_size": float(
            cluster_sizes.median()
        ),
        "number_of_mixed_label_clusters": int(
            mixed_mask.sum()
        ),
        "sequences_in_mixed_label_clusters": int(
            cluster_summary.loc[
                mixed_mask,
                "cluster_size",
            ].sum()
        ),
        "compression_ratio_clusters_per_sequence": float(
            len(cluster_summary)
            / len(membership)
        ),
    }

    return (
        membership,
        cluster_summary,
        global_summary,
    )


def write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Write JSON using deterministic formatting."""
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


def file_record(
    path: Path,
) -> dict[str, Any]:
    """Build a manifest record for an output file."""
    return {
        "path": str(path),
        "size_bytes": (
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def run_one_configuration(
    *,
    dataframe: pd.DataFrame,
    input_path: Path,
    output_root: Path,
    config: dict[str, Any],
    task_id: str,
    config_id: str,
    config_definition: dict[str, Any],
    mmseqs_binary: str,
    mmseqs_version: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Run one task/configuration clustering job."""
    dataset_config = config["dataset"]
    mmseqs_config = config["mmseqs"]
    output_config = config.get(
        "output",
        {},
    )

    task_column = dataset_config[
        "task_column"
    ]
    sequence_id_column = dataset_config[
        "sequence_id_column"
    ]
    sequence_column = dataset_config[
        "sequence_column"
    ]
    label_column = dataset_config[
        "label_column"
    ]

    run_directory = (
        output_root
        / task_id
        / config_id
    )

    if run_directory.exists():
        if overwrite:
            shutil.rmtree(
                run_directory
            )
        else:
            raise FileExistsError(
                "Output directory already exists: "
                f"{run_directory}. "
                "Use --overwrite to replace it."
            )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    task_dataframe = dataframe[
        dataframe[task_column]
        == task_id
    ].copy()

    task_dataframe = task_dataframe.sort_values(
        sequence_id_column
    ).reset_index(drop=True)

    fasta_path = (
        run_directory
        / "sequences.fasta"
    )
    raw_tsv_path = (
        run_directory
        / "mmseqs_clusters.tsv"
    )
    membership_path = (
        run_directory
        / "cluster_membership.csv"
    )
    cluster_summary_path = (
        run_directory
        / "cluster_summary.csv"
    )
    global_summary_path = (
        run_directory
        / "clustering_summary.json"
    )
    effective_config_path = (
        run_directory
        / "effective_config.yml"
    )
    log_path = (
        run_directory
        / "mmseqs.log"
    )
    manifest_path = (
        run_directory
        / "manifest.json"
    )
    database_directory = (
        run_directory
        / "mmseqs_db"
    )
    temporary_directory = (
        run_directory
        / "tmp"
    )

    database_directory.mkdir()
    temporary_directory.mkdir()

    sequence_db = (
        database_directory
        / "sequence_db"
    )
    cluster_db = (
        database_directory
        / "cluster_db"
    )

    write_fasta(
        task_dataframe=task_dataframe,
        sequence_id_column=(
            sequence_id_column
        ),
        sequence_column=sequence_column,
        output_path=fasta_path,
    )

    minimum_identity = float(
        config_definition[
            "minimum_sequence_identity"
        ]
    )

    effective_config = {
        "task_id": task_id,
        "homology_config_id": config_id,
        "minimum_sequence_identity": (
            minimum_identity
        ),
        "mmseqs": mmseqs_config,
        "configuration_overrides": (
            config_definition
        ),
    }

    with effective_config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            effective_config,
            handle,
            sort_keys=False,
        )

    environment = os.environ.copy()

    # Disable terminal-dependent progress formatting.
    environment["TTY"] = "0"

    command_records: list[
        dict[str, Any]
    ] = []

    command_records.append(
        run_command(
            [
                mmseqs_binary,
                "createdb",
                str(fasta_path),
                str(sequence_db),
            ],
            log_path,
            environment,
        )
    )

    command_records.append(
        run_command(
            build_cluster_command(
                binary=mmseqs_binary,
                sequence_db=sequence_db,
                cluster_db=cluster_db,
                temporary_directory=(
                    temporary_directory
                ),
                mmseqs_config=(
                    mmseqs_config
                ),
                identity=minimum_identity,
                config_definition=(
                    config_definition
                ),
            ),
            log_path,
            environment,
        )
    )

    command_records.append(
        run_command(
            [
                mmseqs_binary,
                "createtsv",
                str(sequence_db),
                str(sequence_db),
                str(cluster_db),
                str(raw_tsv_path),
            ],
            log_path,
            environment,
        )
    )

    expected_sequence_ids = set(
        task_dataframe[
            sequence_id_column
        ].astype(str)
    )

    clusters = parse_cluster_tsv(
        tsv_path=raw_tsv_path,
        expected_sequence_ids=(
            expected_sequence_ids
        ),
    )

    (
        membership,
        cluster_summary,
        global_summary,
    ) = build_cluster_tables(
        task_dataframe=task_dataframe,
        clusters=clusters,
        task_id=task_id,
        config_id=config_id,
        identity=minimum_identity,
        mmseqs_config=mmseqs_config,
        dataset_config=dataset_config,
        hash_prefix_length=int(
            output_config.get(
                "cluster_hash_prefix_length",
                16,
            )
        ),
    )

    membership.to_csv(
        membership_path,
        index=False,
    )

    cluster_summary.to_csv(
        cluster_summary_path,
        index=False,
    )

    write_json(
        global_summary,
        global_summary_path,
    )

    if not output_config.get(
        "keep_mmseqs_databases",
        False,
    ):
        shutil.rmtree(
            database_directory
        )

    if not output_config.get(
        "keep_temporary_directory",
        False,
    ):
        shutil.rmtree(
            temporary_directory
        )

    output_files = [
        fasta_path,
        raw_tsv_path,
        membership_path,
        cluster_summary_path,
        global_summary_path,
        effective_config_path,
        log_path,
    ]

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "task_id": task_id,
        "homology_config_id": (
            config_id
        ),
        "operational_definition": (
            "MMseqs2 sequence-similarity connected components "
            "generated from all positive and background sequences "
            "within the task."
        ),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(
                input_path
            ),
            "task_rows": int(
                len(task_dataframe)
            ),
            "positive_sequences": int(
                (
                    task_dataframe[
                        label_column
                    ]
                    == 1
                ).sum()
            ),
            "negative_sequences": int(
                (
                    task_dataframe[
                        label_column
                    ]
                    == 0
                ).sum()
            ),
        },
        "software": {
            "mmseqs_binary": (
                mmseqs_binary
            ),
            "mmseqs_version": (
                mmseqs_version
            ),
            "python_version": (
                sys.version
            ),
            "platform": (
                platform.platform()
            ),
        },
        "parameters": {
            "minimum_sequence_identity": (
                minimum_identity
            ),
            "minimum_coverage": float(
                mmseqs_config[
                    "minimum_coverage"
                ]
            ),
            "coverage_mode": int(
                mmseqs_config[
                    "coverage_mode"
                ]
            ),
            "cluster_mode": int(
                mmseqs_config[
                    "cluster_mode"
                ]
            ),
            "alignment_mode": int(
                mmseqs_config[
                    "alignment_mode"
                ]
            ),
            "threads": int(
                mmseqs_config["threads"]
            ),
            "e_value": (
                config_definition.get(
                    "e_value",
                    mmseqs_config.get(
                        "e_value"
                    ),
                )
            ),
            "sensitivity": (
                config_definition.get(
                    "sensitivity",
                    mmseqs_config.get(
                        "sensitivity"
                    ),
                )
            ),
            "max_sequences": (
                config_definition.get(
                    "max_sequences",
                    mmseqs_config.get(
                        "max_sequences"
                    ),
                )
            ),
            "extra_args": [
                *mmseqs_config.get(
                    "extra_args",
                    [],
                ),
                *config_definition.get(
                    "extra_args",
                    [],
                ),
            ],
        },
        "commands": command_records,
        "clustering_summary": (
            global_summary
        ),
        "outputs": [
            file_record(path)
            for path in output_files
        ],
    }

    write_json(
        manifest,
        manifest_path,
    )

    return {
        "task_id": task_id,
        "homology_config_id": (
            config_id
        ),
        **global_summary,
        "manifest": str(
            manifest_path
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run task-specific MMseqs2 clustering at multiple "
            "sequence-identity thresholds and export stable "
            "cluster memberships."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to data/processed/amp_task.csv.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/homology_clustering.yml.",
    )

    parser.add_argument(
        "--output-root",
        default="data/homology",
        help=(
            "Root directory for task/configuration outputs. "
            "Default: data/homology"
        ),
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Optional task IDs to run. "
            "Defaults to all configured tasks."
        ),
    )

    parser.add_argument(
        "--configurations",
        nargs="+",
        default=None,
        help=(
            "Optional configurations to run, "
            "for example H90 H70."
        ),
    )

    parser.add_argument(
        "--mmseqs-binary",
        default=None,
        help=(
            "Optional override for the MMseqs2 executable."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing task/configuration output directories."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run all requested task/configuration combinations."""
    args = parse_arguments()

    input_path = Path(args.input)
    config_path = Path(args.config)
    output_root = Path(
        args.output_root
    )

    try:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                "Configuration file not found: "
                f"{config_path}"
            )

        config = load_config(
            config_path
        )

        validate_config(
            config
        )

        task_filter = (
            set(args.tasks)
            if args.tasks
            else None
        )

        dataframe = load_dataset(
            input_path=input_path,
            config=config,
            task_filter=task_filter,
        )

        task_column = config[
            "dataset"
        ]["task_column"]

        available_tasks = sorted(
            dataframe[
                task_column
            ].astype(str).unique()
        )

        requested_configurations = (
            set(args.configurations)
            if args.configurations
            else None
        )

        enabled_configurations = {
            str(config_id): definition
            for config_id, definition
            in config[
                "configurations"
            ].items()
            if definition.get(
                "enabled",
                True,
            )
            and (
                requested_configurations
                is None
                or str(config_id)
                in requested_configurations
            )
        }

        if requested_configurations:
            missing_configurations = (
                requested_configurations
                - set(
                    enabled_configurations
                )
            )

            if missing_configurations:
                raise ValueError(
                    "Requested configurations are missing "
                    "or disabled: "
                    + ", ".join(
                        sorted(
                            missing_configurations
                        )
                    )
                )

        mmseqs_binary = (
            resolve_mmseqs_binary(
                configured=str(
                    config[
                        "mmseqs"
                    ]["binary"]
                ),
                override=(
                    args.mmseqs_binary
                ),
            )
        )

        mmseqs_version = (
            get_mmseqs_version(
                mmseqs_binary
            )
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_summaries: list[
            dict[str, Any]
        ] = []

        for task_id in available_tasks:
            for (
                config_id,
                definition,
            ) in enabled_configurations.items():
                print(
                    f"Running {task_id} / {config_id}"
                )

                summary = (
                    run_one_configuration(
                        dataframe=dataframe,
                        input_path=(
                            input_path
                        ),
                        output_root=(
                            output_root
                        ),
                        config=config,
                        task_id=task_id,
                        config_id=config_id,
                        config_definition=(
                            definition
                        ),
                        mmseqs_binary=(
                            mmseqs_binary
                        ),
                        mmseqs_version=(
                            mmseqs_version
                        ),
                        overwrite=(
                            args.overwrite
                        ),
                    )
                )

                run_summaries.append(
                    summary
                )

                print(
                    "  "
                    f"clusters="
                    f"{summary['number_of_clusters']} "
                    f"singletons="
                    f"{summary['number_of_singletons']} "
                    f"largest="
                    f"{summary['largest_cluster_size']}"
                )

        run_index_path = (
            output_root
            / "clustering_run_index.csv"
        )

        pd.DataFrame(
            run_summaries
        ).sort_values(
            [
                "task_id",
                "homology_config_id",
            ]
        ).to_csv(
            run_index_path,
            index=False,
        )

        print()
        print(
            f"MMseqs2 version: "
            f"{mmseqs_version}"
        )
        print(
            f"Completed runs: "
            f"{len(run_summaries)}"
        )
        print(
            f"Run index: "
            f"{run_index_path}"
        )

        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
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