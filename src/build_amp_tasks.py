#!/usr/bin/env python3

"""Build reproducible AMP classification tasks from a normalized CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_NEGATIVE_MODES = {
    "toxic_without_amp_evidence",
    "all_without_amp_evidence",
}


def decode_separator(value: str) -> str:
    """Convert escaped separators such as '\\t' to actual characters."""
    separators = {
        r"\t": "\t",
        r"\n": "\n",
        r"\r": "\r",
    }

    separator = separators.get(value, value)

    if len(separator) != 1:
        raise ValueError(
            "The separator must resolve to exactly one character."
        )

    return separator


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def normalize_scalar(value: Any) -> str:
    """
    Normalize scalar values for deterministic comparison.

    Examples:
        1, "1", and "1.0" become "1".
        " Positive " becomes "positive".
    """
    if value is None:
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    try:
        numeric_value = Decimal(text)

        if numeric_value == numeric_value.to_integral():
            return str(numeric_value.to_integral())

        return format(numeric_value.normalize(), "f")

    except InvalidOperation:
        return text.lower()


def normalize_missing_values(values: list[Any]) -> set[str]:
    """Normalize the configured representations of missing values."""
    return {
        normalize_scalar(value)
        for value in values
    }


def parse_boolean(value: Any) -> bool:
    """Interpret common boolean representations."""
    normalized = normalize_scalar(value)

    if normalized in {"1", "true", "yes", "y"}:
        return True

    if normalized in {"0", "false", "no", "n", ""}:
        return False

    raise ValueError(
        f"Cannot interpret value as boolean: {value!r}"
    )


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate the principal task configuration fields."""
    dataset_config = config.get("dataset", {})
    labels_config = config.get("labels", {})
    tasks_config = config.get("tasks", {})

    required_dataset_fields = [
        "sequence_column",
        "sequence_id_column",
        "sequence_length_column",
        "valid_flag_column",
        "antimicrobial_column",
    ]

    missing_dataset_fields = [
        field
        for field in required_dataset_fields
        if not dataset_config.get(field)
    ]

    if missing_dataset_fields:
        missing = ", ".join(missing_dataset_fields)
        raise ValueError(
            "Missing required dataset configuration fields: "
            f"{missing}"
        )

    toxicity_columns = dataset_config.get(
        "toxicity_columns",
        [],
    )

    if not isinstance(toxicity_columns, list):
        raise ValueError(
            "dataset.toxicity_columns must be a YAML list."
        )

    required_label_fields = [
        "antimicrobial_positive_value",
        "antimicrobial_negative_value",
        "toxicity_positive_value",
    ]

    missing_label_fields = [
        field
        for field in required_label_fields
        if field not in labels_config
    ]

    if missing_label_fields:
        missing = ", ".join(missing_label_fields)
        raise ValueError(
            "Missing required label configuration fields: "
            f"{missing}"
        )

    if not isinstance(tasks_config, dict) or not tasks_config:
        raise ValueError(
            "At least one task must be defined under tasks."
        )

    enabled_tasks = 0

    for task_id, task_config in tasks_config.items():
        if not isinstance(task_config, dict):
            raise ValueError(
                f"Task '{task_id}' must contain a YAML mapping."
            )

        if not task_config.get("enabled", True):
            continue

        enabled_tasks += 1
        negative_mode = task_config.get("negative_mode")

        if negative_mode not in SUPPORTED_NEGATIVE_MODES:
            supported = ", ".join(
                sorted(SUPPORTED_NEGATIVE_MODES)
            )
            raise ValueError(
                f"Task '{task_id}' has unsupported negative_mode "
                f"'{negative_mode}'. Supported modes: {supported}"
            )

        if (
            negative_mode == "toxic_without_amp_evidence"
            and not toxicity_columns
        ):
            raise ValueError(
                f"Task '{task_id}' requires at least one "
                "toxicity column."
            )

    if enabled_tasks == 0:
        raise ValueError(
            "The configuration does not contain any enabled tasks."
        )


def validate_input_columns(
    columns: list[str],
    config: dict[str, Any],
) -> None:
    """Check that all configured input columns are available."""
    dataset_config = config["dataset"]

    required_columns = [
        dataset_config["sequence_column"],
        dataset_config["sequence_id_column"],
        dataset_config["sequence_length_column"],
        dataset_config["valid_flag_column"],
        dataset_config["antimicrobial_column"],
        *dataset_config.get("toxicity_columns", []),
    ]

    raw_row_id_column = dataset_config.get(
        "raw_row_id_column"
    )

    if raw_row_id_column:
        required_columns.append(raw_row_id_column)

    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Required input columns were not found: {missing}"
        )


def detect_toxicity_evidence(
    row: dict[str, str],
    toxicity_columns: list[str],
    toxicity_positive_value: str,
    missing_values: set[str],
) -> list[str]:
    """Return toxicity columns with a configured positive value."""
    evidence: list[str] = []

    for column in toxicity_columns:
        value = normalize_scalar(row.get(column))

        if value in missing_values:
            continue

        if value == toxicity_positive_value:
            evidence.append(column)

    return evidence


def build_output_row(
    *,
    task_id: str,
    sequence_id: str,
    sequence: str,
    raw_row_id: str,
    sequence_length: int,
    canonical_sequence: bool,
    amp_label: int,
    class_definition: str,
    positive_definition: str,
    negative_definition: str,
    antimicrobial_source_value: str,
    toxicity_evidence: list[str],
) -> dict[str, Any]:
    """Build one task-specific output record."""
    return {
        "task_id": task_id,
        "sequence_id": sequence_id,
        "sequence": sequence,
        "amp_label": amp_label,
        "class_definition": class_definition,
        "positive_definition": positive_definition,
        "negative_definition": negative_definition,
        "sequence_length": sequence_length,
        "canonical_sequence": int(canonical_sequence),
        "raw_row_id": raw_row_id,
        "antimicrobial_source_value": (
            antimicrobial_source_value
        ),
        "toxicity_positive": int(bool(toxicity_evidence)),
        "toxicity_evidence": "|".join(
            sorted(toxicity_evidence)
        ),
    }


def build_amp_tasks(
    input_path: Path,
    config: dict[str, Any],
    encoding_override: str | None = None,
    separator_override: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all configured AMP tasks."""
    input_config = config.get("input", {})
    dataset_config = config["dataset"]
    labels_config = config["labels"]
    tasks_config = config["tasks"]

    encoding = (
        encoding_override
        or input_config.get("encoding", "utf-8")
    )

    separator_value = (
        separator_override
        if separator_override is not None
        else input_config.get("separator", ",")
    )
    separator = decode_separator(separator_value)

    sequence_column = dataset_config["sequence_column"]
    sequence_id_column = dataset_config[
        "sequence_id_column"
    ]
    sequence_length_column = dataset_config[
        "sequence_length_column"
    ]
    valid_flag_column = dataset_config["valid_flag_column"]
    antimicrobial_column = dataset_config[
        "antimicrobial_column"
    ]

    raw_row_id_column = dataset_config.get(
        "raw_row_id_column"
    )
    toxicity_columns = dataset_config.get(
        "toxicity_columns",
        [],
    )

    require_valid_sequences = dataset_config.get(
        "require_valid_sequences",
        True,
    )
    require_unique_sequence_ids = dataset_config.get(
        "require_unique_sequence_ids",
        True,
    )

    amp_positive_value = normalize_scalar(
        labels_config["antimicrobial_positive_value"]
    )
    amp_negative_value = normalize_scalar(
        labels_config["antimicrobial_negative_value"]
    )
    toxicity_positive_value = normalize_scalar(
        labels_config["toxicity_positive_value"]
    )

    missing_values = normalize_missing_values(
        labels_config.get(
            "missing_values",
            ["", "NA", "NaN", "null", "None"],
        )
    )

    enabled_tasks = {
        task_id: task_config
        for task_id, task_config in tasks_config.items()
        if task_config.get("enabled", True)
    }

    output_rows: list[dict[str, Any]] = []
    observed_sequence_ids: set[str] = set()

    report_counters = Counter()
    task_counters: dict[str, Counter] = {
        task_id: Counter()
        for task_id in enabled_tasks
    }

    with input_path.open(
        "r",
        encoding=encoding,
        newline="",
    ) as input_handle:
        reader = csv.DictReader(
            input_handle,
            delimiter=separator,
        )

        if reader.fieldnames is None:
            raise ValueError(
                "The input CSV does not contain a header."
            )

        validate_input_columns(
            columns=list(reader.fieldnames),
            config=config,
        )

        for source_row_number, row in enumerate(
            reader,
            start=2,
        ):
            report_counters["input_rows"] += 1

            sequence = (row.get(sequence_column) or "").strip()
            sequence_id = (
                row.get(sequence_id_column) or ""
            ).strip()

            if not sequence:
                report_counters[
                    "excluded_empty_sequence"
                ] += 1
                continue

            if not sequence_id:
                report_counters[
                    "excluded_missing_sequence_id"
                ] += 1
                continue

            if require_unique_sequence_ids:
                if sequence_id in observed_sequence_ids:
                    raise ValueError(
                        "Duplicated sequence_id detected at "
                        f"CSV row {source_row_number}: "
                        f"{sequence_id}"
                    )

                observed_sequence_ids.add(sequence_id)

            try:
                is_valid = parse_boolean(
                    row.get(valid_flag_column)
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid validity flag at CSV row "
                    f"{source_row_number}: "
                    f"{row.get(valid_flag_column)!r}"
                ) from error

            if require_valid_sequences and not is_valid:
                report_counters[
                    "excluded_invalid_sequence"
                ] += 1
                continue

            try:
                sequence_length = int(
                    row[sequence_length_column]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid sequence length at CSV row "
                    f"{source_row_number}: "
                    f"{row.get(sequence_length_column)!r}"
                ) from error

            antimicrobial_source_value = normalize_scalar(
                row.get(antimicrobial_column)
            )

            if (
                antimicrobial_source_value
                in missing_values
            ):
                report_counters[
                    "excluded_missing_amp_label"
                ] += 1
                continue

            if antimicrobial_source_value == amp_positive_value:
                is_amp_positive = True
                is_amp_negative = False

            elif (
                antimicrobial_source_value
                == amp_negative_value
            ):
                is_amp_positive = False
                is_amp_negative = True

            else:
                report_counters[
                    "excluded_unexpected_amp_label"
                ] += 1
                continue

            toxicity_evidence = detect_toxicity_evidence(
                row=row,
                toxicity_columns=toxicity_columns,
                toxicity_positive_value=(
                    toxicity_positive_value
                ),
                missing_values=missing_values,
            )

            if raw_row_id_column:
                raw_row_id = (
                    row.get(raw_row_id_column) or ""
                ).strip()

                if not raw_row_id:
                    raw_row_id = sequence_id
            else:
                # The source contains one row per unique sequence.
                raw_row_id = sequence_id

            report_counters["eligible_rows"] += 1

            for task_id, task_config in enabled_tasks.items():
                negative_mode = task_config[
                    "negative_mode"
                ]
                positive_definition = task_config.get(
                    "positive_definition",
                    "antimicrobial_activity_reported",
                )
                negative_definition = task_config.get(
                    "negative_definition",
                    "not_defined",
                )

                if is_amp_positive:
                    output_rows.append(
                        build_output_row(
                            task_id=task_id,
                            sequence_id=sequence_id,
                            sequence=sequence,
                            raw_row_id=raw_row_id,
                            sequence_length=sequence_length,
                            canonical_sequence=is_valid,
                            amp_label=1,
                            class_definition=(
                                "strict_positive"
                            ),
                            positive_definition=(
                                positive_definition
                            ),
                            negative_definition=(
                                "not_applicable"
                            ),
                            antimicrobial_source_value=(
                                antimicrobial_source_value
                            ),
                            toxicity_evidence=(
                                toxicity_evidence
                            ),
                        )
                    )

                    task_counters[task_id][
                        "positive_sequences"
                    ] += 1
                    continue

                include_negative = False

                if (
                    negative_mode
                    == "toxic_without_amp_evidence"
                ):
                    include_negative = bool(
                        toxicity_evidence
                    )

                elif (
                    negative_mode
                    == "all_without_amp_evidence"
                ):
                    include_negative = is_amp_negative

                if include_negative:
                    output_rows.append(
                        build_output_row(
                            task_id=task_id,
                            sequence_id=sequence_id,
                            sequence=sequence,
                            raw_row_id=raw_row_id,
                            sequence_length=sequence_length,
                            canonical_sequence=is_valid,
                            amp_label=0,
                            class_definition=(
                                "candidate_negative_background"
                            ),
                            positive_definition=(
                                positive_definition
                            ),
                            negative_definition=(
                                negative_definition
                            ),
                            antimicrobial_source_value=(
                                antimicrobial_source_value
                            ),
                            toxicity_evidence=(
                                toxicity_evidence
                            ),
                        )
                    )

                    task_counters[task_id][
                        "negative_sequences"
                    ] += 1

                    if toxicity_evidence:
                        task_counters[task_id][
                            "toxic_negative_sequences"
                        ] += 1
                else:
                    task_counters[task_id][
                        "excluded_by_task_definition"
                    ] += 1

    # Deterministic output independent of input row order.
    output_rows.sort(
        key=lambda record: (
            record["task_id"],
            record["sequence_id"],
        )
    )

    tasks_report: dict[str, Any] = {}

    for task_id, counters in task_counters.items():
        positives = counters["positive_sequences"]
        negatives = counters["negative_sequences"]
        total = positives + negatives

        prevalence = (
            positives / total
            if total > 0
            else None
        )

        tasks_report[task_id] = {
            "negative_mode": enabled_tasks[task_id][
                "negative_mode"
            ],
            "positive_definition": enabled_tasks[
                task_id
            ].get(
                "positive_definition",
                "antimicrobial_activity_reported",
            ),
            "negative_definition": enabled_tasks[
                task_id
            ].get(
                "negative_definition",
                "not_defined",
            ),
            "total_sequences": total,
            "positive_sequences": positives,
            "negative_sequences": negatives,
            "positive_prevalence": prevalence,
            "toxic_negative_sequences": counters[
                "toxic_negative_sequences"
            ],
            "excluded_by_task_definition": counters[
                "excluded_by_task_definition"
            ],
        }

    report = {
        "input_file": str(input_path),
        "encoding": encoding,
        "separator": (
            r"\t"
            if separator == "\t"
            else separator
        ),
        "antimicrobial_column": antimicrobial_column,
        "toxicity_columns": toxicity_columns,
        "raw_row_id_strategy": (
            f"input_column:{raw_row_id_column}"
            if raw_row_id_column
            else "sequence_id_fallback"
        ),
        "input_summary": {
            "input_rows": report_counters["input_rows"],
            "eligible_rows": report_counters[
                "eligible_rows"
            ],
            "excluded_empty_sequence": report_counters[
                "excluded_empty_sequence"
            ],
            "excluded_missing_sequence_id": (
                report_counters[
                    "excluded_missing_sequence_id"
                ]
            ),
            "excluded_invalid_sequence": report_counters[
                "excluded_invalid_sequence"
            ],
            "excluded_missing_amp_label": report_counters[
                "excluded_missing_amp_label"
            ],
            "excluded_unexpected_amp_label": (
                report_counters[
                    "excluded_unexpected_amp_label"
                ]
            ),
        },
        "tasks": tasks_report,
        "output_rows": len(output_rows),
    }

    return output_rows, report


def write_task_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write all AMP tasks in deterministic long format."""
    fieldnames = [
        "task_id",
        "sequence_id",
        "sequence",
        "amp_label",
        "class_definition",
        "positive_definition",
        "negative_definition",
        "sequence_length",
        "canonical_sequence",
        "raw_row_id",
        "antimicrobial_source_value",
        "toxicity_positive",
        "toxicity_evidence",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the task-construction report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        json.dump(
            report,
            output_handle,
            indent=2,
            ensure_ascii=False,
        )
        output_handle.write("\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build AMP classification tasks from a normalized "
            "sequence CSV using a YAML configuration."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the normalized sequence CSV.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the AMP task YAML configuration.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/amp_task.csv",
        help=(
            "Path to the combined task CSV. "
            "Default: data/processed/amp_task.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default="metadata/amp_task_report.json",
        help=(
            "Path to the JSON construction report. "
            "Default: metadata/amp_task_report.json"
        ),
    )

    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional encoding override.",
    )

    parser.add_argument(
        "--separator",
        default=None,
        help=r"Optional separator override, such as ',' or '\t'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )

    return parser.parse_args()


def main() -> int:
    """Build and export the configured AMP tasks."""
    args = parse_arguments()

    input_path = Path(args.input)
    config_path = Path(args.config)
    output_path = Path(args.output)
    report_path = Path(args.report)

    try:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        if not args.overwrite:
            existing_files = [
                path
                for path in [output_path, report_path]
                if path.exists()
            ]

            if existing_files:
                existing = ", ".join(
                    str(path)
                    for path in existing_files
                )
                raise FileExistsError(
                    f"Output file already exists: {existing}. "
                    "Use --overwrite to replace it."
                )

        config = load_config(config_path)
        validate_configuration(config)

        rows, report = build_amp_tasks(
            input_path=input_path,
            config=config,
            encoding_override=args.encoding,
            separator_override=args.separator,
        )

        if not rows:
            raise ValueError(
                "No sequences satisfied the configured task "
                "definitions."
            )

        write_task_csv(
            rows=rows,
            output_path=output_path,
        )
        write_json_report(
            report=report,
            output_path=report_path,
        )

        print(f"Task dataset written to: {output_path}")
        print(f"Construction report written to: {report_path}")
        print(f"Output rows: {report['output_rows']}")
        print()

        for task_id, task_report in report["tasks"].items():
            print(task_id)
            print(
                f"  total: "
                f"{task_report['total_sequences']}"
            )
            print(
                f"  positives: "
                f"{task_report['positive_sequences']}"
            )
            print(
                f"  negatives: "
                f"{task_report['negative_sequences']}"
            )
            print(
                f"  positive prevalence: "
                f"{task_report['positive_prevalence']}"
            )

        return 0

    except (
        OSError,
        ValueError,
        csv.Error,
        yaml.YAMLError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())