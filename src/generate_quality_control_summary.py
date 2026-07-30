#!/usr/bin/env python3

"""Generate a quality-control summary from a normalized sequence CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def decode_separator(value: str) -> str:
    """Convert escaped separators such as '\\t' into actual characters."""
    separators = {
        r"\t": "\t",
        r"\n": "\n",
        r"\r": "\r",
    }

    separator = separators.get(value, value)

    if len(separator) != 1:
        raise ValueError(
            "The separator must be exactly one character, "
            "for example ',', ';', or '\\t'."
        )

    return separator


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


def parse_boolean_flag(value: str | None) -> bool:
    """Interpret common CSV boolean representations."""
    if value is None:
        return False

    normalized = str(value).strip().lower()

    return normalized in {
        "1",
        "true",
        "yes",
        "y",
    }


def normalize_label_value(value: str | None) -> str:
    """Normalize a label value for deterministic comparison."""
    if value is None:
        return ""

    return str(value).strip()


def validate_columns(
    columns: list[str],
    sequence_column: str,
    label_columns: list[str],
    required_derived_columns: list[str],
) -> None:
    """Validate the columns required for quality-control calculations."""
    required_columns = [
        sequence_column,
        *required_derived_columns,
        *label_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"The following required columns were not found: {missing}"
        )


def calculate_quality_control(
    input_path: Path,
    config: dict[str, Any],
    encoding_override: str | None = None,
    separator_override: str | None = None,
) -> dict[str, Any]:
    """Calculate dataset-level quality-control statistics."""
    input_config = config.get("input", {})
    dataset_config = config.get("dataset", {})
    output_config = config.get("output", {})

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

    sequence_column = dataset_config.get("sequence_column")
    label_columns = dataset_config.get("label_columns", [])

    if not sequence_column:
        raise ValueError(
            "dataset.sequence_column must be defined in the configuration."
        )

    if label_columns is None:
        label_columns = []

    if not isinstance(label_columns, list):
        raise ValueError(
            "dataset.label_columns must be defined as a YAML list."
        )

    sequence_length_column = output_config.get(
        "sequence_length_column",
        "sequence_length",
    )
    empty_flag_column = output_config.get(
        "empty_flag_column",
        "sequence_is_empty",
    )
    valid_flag_column = output_config.get(
        "valid_flag_column",
        "sequence_is_valid",
    )
    invalid_characters_column = output_config.get(
        "invalid_characters_column",
        "invalid_characters",
    )

    required_derived_columns = [
        sequence_length_column,
        empty_flag_column,
        valid_flag_column,
        invalid_characters_column,
    ]

    total_rows = 0
    valid_sequences = 0
    empty_sequences = 0
    noncanonical_sequences = 0

    normalized_sequences: list[str] = []
    valid_lengths: list[int] = []

    # Each normalized sequence may theoretically appear more than once
    # after case conversion and whitespace/format removal.
    sequence_label_profiles: dict[
        str,
        set[tuple[str, ...]],
    ] = defaultdict(set)

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
            raise ValueError("The input CSV does not contain a header.")

        columns = list(reader.fieldnames)

        validate_columns(
            columns=columns,
            sequence_column=sequence_column,
            label_columns=label_columns,
            required_derived_columns=required_derived_columns,
        )

        for row in reader:
            total_rows += 1

            sequence = row.get(sequence_column, "")
            sequence = "" if sequence is None else sequence

            is_empty = parse_boolean_flag(
                row.get(empty_flag_column)
            )
            is_valid = parse_boolean_flag(
                row.get(valid_flag_column)
            )

            invalid_characters = (
                row.get(invalid_characters_column, "") or ""
            )

            if is_empty:
                empty_sequences += 1

            if is_valid:
                valid_sequences += 1

                try:
                    sequence_length = int(
                        row[sequence_length_column]
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Invalid sequence length at input row "
                        f"{total_rows + 1}: "
                        f"{row.get(sequence_length_column)!r}"
                    ) from error

                valid_lengths.append(sequence_length)

            if not is_empty and invalid_characters:
                noncanonical_sequences += 1

            if sequence:
                normalized_sequences.append(sequence)

                label_profile = tuple(
                    normalize_label_value(row.get(column))
                    for column in label_columns
                )

                sequence_label_profiles[sequence].add(
                    label_profile
                )

    unique_sequences = len(set(normalized_sequences))

    # Number of additional rows sharing an already observed normalized
    # sequence. Example: three occurrences of one sequence count as two
    # exact duplicates.
    exact_duplicates = (
        len(normalized_sequences) - unique_sequences
    )

    # Number of normalized sequences associated with more than one
    # label profile.
    label_conflicts = sum(
        1
        for profiles in sequence_label_profiles.values()
        if len(profiles) > 1
    )

    if valid_lengths:
        minimum_length: int | None = min(valid_lengths)
        maximum_length: int | None = max(valid_lengths)
        median_length: float | int | None = statistics.median(
            valid_lengths
        )
    else:
        minimum_length = None
        maximum_length = None
        median_length = None

    return {
        "input_file": str(input_path),
        "definitions": {
            "unique_sequences": (
                "Number of distinct non-empty normalized sequences."
            ),
            "exact_duplicates": (
                "Number of additional rows sharing an identical "
                "non-empty normalized sequence."
            ),
            "noncanonical_sequences": (
                "Number of non-empty sequences containing at least "
                "one character outside the configured alphabet."
            ),
            "label_conflicts": (
                "Number of duplicated normalized sequences associated "
                "with more than one label profile."
            ),
            "length_statistics": (
                "Calculated using valid, non-empty sequences only."
            ),
        },
        "controls": {
            "original_rows": total_rows,
            "valid_sequences": valid_sequences,
            "unique_sequences": unique_sequences,
            "exact_duplicates": exact_duplicates,
            "noncanonical_sequences": noncanonical_sequences,
            "empty_sequences": empty_sequences,
            "label_conflicts": label_conflicts,
            "minimum_length": minimum_length,
            "maximum_length": maximum_length,
            "median_length": median_length,
        },
        "label_columns_checked": label_columns,
        "encoding": encoding,
        "separator": r"\t" if separator == "\t" else separator,
    }


def write_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the complete quality-control report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_handle:
        json.dump(
            report,
            output_handle,
            indent=2,
            ensure_ascii=False,
        )
        output_handle.write("\n")


def write_csv_summary(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the quality-control controls as a two-column CSV table."""
    display_names = {
        "original_rows": "Original rows",
        "valid_sequences": "Valid sequences",
        "unique_sequences": "Unique sequences",
        "exact_duplicates": "Exact duplicates",
        "noncanonical_sequences": "Non-canonical sequences",
        "empty_sequences": "Empty sequences",
        "label_conflicts": "Label conflicts",
        "minimum_length": "Minimum length",
        "maximum_length": "Maximum length",
        "median_length": "Median length",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["control", "result"],
        )
        writer.writeheader()

        for key, value in report["controls"].items():
            writer.writerow(
                {
                    "control": display_names[key],
                    "result": "" if value is None else value,
                }
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a quality-control summary from a normalized "
            "sequence CSV."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the normalized CSV file.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--json-output",
        default="metadata/quality_control_summary.json",
        help=(
            "Path to the JSON quality-control report. "
            "Default: metadata/quality_control_summary.json"
        ),
    )

    parser.add_argument(
        "--csv-output",
        default="metadata/quality_control_summary.csv",
        help=(
            "Path to the CSV quality-control table. "
            "Default: metadata/quality_control_summary.csv"
        ),
    )

    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional override for the configured encoding.",
    )

    parser.add_argument(
        "--separator",
        default=None,
        help=r"Optional separator override, such as ',' or '\t'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing report files to be replaced.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the quality-control summary generation."""
    args = parse_arguments()

    input_path = Path(args.input)
    config_path = Path(args.config)
    json_output_path = Path(args.json_output)
    csv_output_path = Path(args.csv_output)

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
            existing_outputs = [
                path
                for path in [
                    json_output_path,
                    csv_output_path,
                ]
                if path.exists()
            ]

            if existing_outputs:
                files = ", ".join(
                    str(path)
                    for path in existing_outputs
                )
                raise FileExistsError(
                    f"Output file already exists: {files}. "
                    "Use --overwrite to replace it."
                )

        config = load_config(config_path)

        report = calculate_quality_control(
            input_path=input_path,
            config=config,
            encoding_override=args.encoding,
            separator_override=args.separator,
        )

        write_json_report(
            report=report,
            output_path=json_output_path,
        )

        write_csv_summary(
            report=report,
            output_path=csv_output_path,
        )

        controls = report["controls"]

        print(f"JSON report written to: {json_output_path}")
        print(f"CSV summary written to: {csv_output_path}")
        print()
        print("Quality-control summary")
        print("-----------------------")

        for control, result in controls.items():
            print(f"{control}: {result}")

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