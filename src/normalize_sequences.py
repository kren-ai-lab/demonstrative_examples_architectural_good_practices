#!/usr/bin/env python3

"""Normalize and validate peptide or protein sequences from a CSV file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def decode_separator(value: str) -> str:
    """Convert common escaped separators to their actual character."""
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
    """Load the YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


def normalize_sequence(
    sequence: str,
    uppercase: bool,
    remove_whitespace: bool,
    remove_format_characters: bool,
) -> str:
    """Apply deterministic sequence normalization."""
    normalized_characters: list[str] = []

    for character in sequence:
        if remove_whitespace and character.isspace():
            continue

        if (
            remove_format_characters
            and unicodedata.category(character) == "Cf"
        ):
            continue

        normalized_characters.append(character)

    normalized = "".join(normalized_characters)

    if uppercase:
        normalized = normalized.upper()

    return normalized


def calculate_sequence_id(sequence: str) -> str:
    """Generate a stable SHA-256 identifier from a normalized sequence."""
    if not sequence:
        return ""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def find_invalid_characters(
    sequence: str,
    alphabet: set[str],
) -> list[str]:
    """Return the sorted set of characters outside the configured alphabet."""
    return sorted(set(sequence) - alphabet)


def validate_output_columns(
    original_columns: list[str],
    derived_columns: list[str],
    sequence_column: str,
) -> None:
    """Check that derived output columns do not overwrite source columns."""
    if sequence_column not in original_columns:
        available_columns = ", ".join(original_columns)
        raise ValueError(
            f"Sequence column '{sequence_column}' was not found. "
            f"Available columns: {available_columns}"
        )

    duplicated_output_columns = [
        column
        for column in derived_columns
        if column in original_columns
    ]

    if duplicated_output_columns:
        duplicated = ", ".join(duplicated_output_columns)
        raise ValueError(
            "The following derived columns already exist in the input CSV: "
            f"{duplicated}"
        )

    if len(set(derived_columns)) != len(derived_columns):
        raise ValueError(
            "Derived column names in the configuration must be unique."
        )


def normalize_csv(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    sequence_column_override: str | None = None,
    encoding_override: str | None = None,
    separator_override: str | None = None,
) -> dict[str, Any]:
    """Normalize sequences and write a processed CSV file."""
    input_config = config.get("input", {})
    dataset_config = config.get("dataset", {})
    normalization_config = config.get("normalization", {})
    output_config = config.get("output", {})

    sequence_column = (
        sequence_column_override
        or dataset_config.get("sequence_column")
    )

    if not sequence_column:
        raise ValueError(
            "A sequence column must be defined in the configuration "
            "or passed with --sequence-column."
        )

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

    uppercase = normalization_config.get("uppercase", True)
    remove_whitespace = normalization_config.get(
        "remove_whitespace",
        True,
    )
    remove_format_characters = normalization_config.get(
        "remove_unicode_format_characters",
        True,
    )

    alphabet_string = normalization_config.get("alphabet")

    if not alphabet_string or not isinstance(alphabet_string, str):
        raise ValueError(
            "normalization.alphabet must be defined as a non-empty string."
        )

    alphabet = set(alphabet_string.upper())

    original_sequence_column = output_config.get(
        "original_sequence_column",
        "sequence_original",
    )
    sequence_id_column = output_config.get(
        "sequence_id_column",
        "sequence_id",
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

    derived_columns = [
        original_sequence_column,
        sequence_id_column,
        sequence_length_column,
        empty_flag_column,
        valid_flag_column,
        invalid_characters_column,
    ]

    total_rows = 0
    valid_sequences = 0
    invalid_sequences = 0
    empty_sequences = 0
    invalid_character_counts: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(
            "r",
            encoding=encoding,
            newline="",
        ) as input_handle,
        output_path.open(
            "w",
            encoding=encoding,
            newline="",
        ) as output_handle,
    ):
        reader = csv.DictReader(
            input_handle,
            delimiter=separator,
        )

        if reader.fieldnames is None:
            raise ValueError("The input CSV does not contain a header.")

        original_columns = list(reader.fieldnames)

        validate_output_columns(
            original_columns=original_columns,
            derived_columns=derived_columns,
            sequence_column=sequence_column,
        )

        output_columns = original_columns + derived_columns

        writer = csv.DictWriter(
            output_handle,
            fieldnames=output_columns,
            delimiter=separator,
            extrasaction="raise",
        )
        writer.writeheader()

        for row in reader:
            total_rows += 1

            raw_sequence = row.get(sequence_column)

            if raw_sequence is None:
                raw_sequence = ""

            normalized_sequence = normalize_sequence(
                sequence=raw_sequence,
                uppercase=uppercase,
                remove_whitespace=remove_whitespace,
                remove_format_characters=remove_format_characters,
            )

            invalid_characters = find_invalid_characters(
                sequence=normalized_sequence,
                alphabet=alphabet,
            )

            is_empty = normalized_sequence == ""
            is_valid = not is_empty and not invalid_characters

            if is_empty:
                empty_sequences += 1

            if is_valid:
                valid_sequences += 1
            else:
                invalid_sequences += 1

            for character in invalid_characters:
                invalid_character_counts[character] += 1

            # Replace the configured sequence column with the normalized value.
            row[sequence_column] = normalized_sequence

            # Preserve the value exactly as it appeared in the input file.
            row[original_sequence_column] = raw_sequence

            row[sequence_id_column] = calculate_sequence_id(
                normalized_sequence
            )
            row[sequence_length_column] = len(normalized_sequence)
            row[empty_flag_column] = int(is_empty)
            row[valid_flag_column] = int(is_valid)
            row[invalid_characters_column] = "".join(
                invalid_characters
            )

            writer.writerow(row)

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "sequence_column": sequence_column,
        "encoding": encoding,
        "separator": r"\t" if separator == "\t" else separator,
        "alphabet": "".join(sorted(alphabet)),
        "normalization": {
            "uppercase": uppercase,
            "remove_whitespace": remove_whitespace,
            "remove_unicode_format_characters": (
                remove_format_characters
            ),
        },
        "number_of_rows": total_rows,
        "number_of_valid_sequences": valid_sequences,
        "number_of_invalid_sequences": invalid_sequences,
        "number_of_empty_sequences": empty_sequences,
        "invalid_character_counts": dict(
            sorted(invalid_character_counts.items())
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and validate sequences in a CSV file using "
            "a YAML configuration."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw CSV file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the normalized CSV file.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--report",
        default="metadata/normalization_report.json",
        help=(
            "Path to the JSON normalization report. "
            "Default: metadata/normalization_report.json"
        ),
    )

    parser.add_argument(
        "--sequence-column",
        default=None,
        help=(
            "Optional override for the sequence column defined "
            "in the YAML configuration."
        ),
    )

    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional override for the input encoding.",
    )

    parser.add_argument(
        "--separator",
        default=None,
        help=r"Optional separator override, such as ',' or '\t'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output and report files to be replaced.",
    )

    return parser.parse_args()


def main() -> int:
    """Run sequence normalization."""
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
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
            if output_path.exists():
                raise FileExistsError(
                    f"Output file already exists: {output_path}. "
                    "Use --overwrite to replace it."
                )

            if report_path.exists():
                raise FileExistsError(
                    f"Report file already exists: {report_path}. "
                    "Use --overwrite to replace it."
                )

        config = load_config(config_path)

        report = normalize_csv(
            input_path=input_path,
            output_path=output_path,
            config=config,
            sequence_column_override=args.sequence_column,
            encoding_override=args.encoding,
            separator_override=args.separator,
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with report_path.open("w", encoding="utf-8") as report_handle:
            json.dump(
                report,
                report_handle,
                indent=2,
                ensure_ascii=False,
            )
            report_handle.write("\n")

        print(f"Normalized CSV written to: {output_path}")
        print(f"Normalization report written to: {report_path}")
        print(f"Rows processed: {report['number_of_rows']}")
        print(
            "Valid sequences: "
            f"{report['number_of_valid_sequences']}"
        )
        print(
            "Invalid sequences: "
            f"{report['number_of_invalid_sequences']}"
        )
        print(
            "Empty sequences: "
            f"{report['number_of_empty_sequences']}"
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