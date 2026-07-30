#!/usr/bin/env python3

"""Generate a metadata manifest for an immutable raw CSV file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_separator(value: str) -> str:
    """Convert escaped separators such as '\\t' into their actual character."""
    decoded = bytes(value, "utf-8").decode("unicode_escape")

    if len(decoded) != 1:
        raise argparse.ArgumentTypeError(
            "The separator must resolve to exactly one character."
        )

    return decoded


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def characterize_csv(
    file_path: Path,
    sequence_column: str,
    encoding: str,
    separator: str,
) -> dict[str, Any]:
    """Count rows, columns, and unique non-empty sequence values."""
    unique_sequences: set[str] = set()

    with file_path.open("r", encoding=encoding, newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter=separator)

        if reader.fieldnames is None:
            raise ValueError("The CSV file does not contain a header.")

        columns = reader.fieldnames

        if sequence_column not in columns:
            available = ", ".join(columns)
            raise ValueError(
                f"Sequence column '{sequence_column}' was not found. "
                f"Available columns: {available}"
            )

        row_count = 0

        for row in reader:
            row_count += 1
            sequence = row.get(sequence_column)

            # Count exact, non-empty values as present in the raw file.
            if sequence is not None and sequence != "":
                unique_sequences.add(sequence)

    return {
        "number_of_rows": row_count,
        "number_of_columns": len(columns),
        "original_columns": columns,
        "number_of_unique_sequences": len(unique_sequences),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build the complete raw-file manifest."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    csv_summary = characterize_csv(
        file_path=input_path,
        sequence_column=args.sequence_column,
        encoding=args.encoding,
        separator=args.separator,
    )

    incorporation_date = args.incorporation_date
    if incorporation_date is None:
        incorporation_date = datetime.now(timezone.utc).isoformat()

    return {
        "file_name": input_path.name,
        "file_size_bytes": input_path.stat().st_size,
        "sha256": calculate_sha256(input_path),
        "incorporation_date": incorporation_date,
        "encoding": args.encoding,
        "separator": "\\t" if args.separator == "\t" else args.separator,
        "sequence_column": args.sequence_column,
        **csv_summary,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSON metadata manifest for a raw CSV file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw CSV file.",
    )

    parser.add_argument(
        "--output",
        default="metadata/raw_file_manifest.json",
        help=(
            "Path to the output JSON manifest. "
            "Default: metadata/raw_file_manifest.json"
        ),
    )

    parser.add_argument(
        "--sequence-column",
        required=True,
        help="Name of the column containing peptide or protein sequences.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding of the CSV file. Default: utf-8",
    )

    parser.add_argument(
        "--separator",
        type=parse_separator,
        default=",",
        help=r"CSV separator. Use ',' for CSV or '\t' for TSV. Default: ','",
    )

    parser.add_argument(
        "--incorporation-date",
        default=None,
        help=(
            "Date when the file was incorporated, preferably in ISO 8601 format. "
            "If omitted, the current UTC timestamp is used."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        manifest = build_manifest(args)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(manifest, file_handle, indent=2, ensure_ascii=False)
            file_handle.write("\n")

        print(f"Manifest written to: {output_path}")
        print(f"SHA-256: {manifest['sha256']}")
        print(f"Rows: {manifest['number_of_rows']}")
        print(f"Unique sequences: {manifest['number_of_unique_sequences']}")

        return 0

    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())