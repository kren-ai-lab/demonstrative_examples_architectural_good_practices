#!/usr/bin/env python3

"""Characterize sequence and physicochemical biases in AMP tasks using modlAMP."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
except ImportError as error:
    raise SystemExit(
        "modlAMP is required. Install it with:\n"
        "pip install modlamp==4.3.2"
    ) from error


PHYSICOCHEMICAL_FEATURES = [
    "net_charge",
    "mean_hydrophobicity",
    "charged_residue_proportion",
    "hydrophobic_ratio",
]


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
            "The separator must resolve to exactly one character."
        )

    return separator


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def get_modlamp_version() -> str:
    """Return the installed modlAMP package version."""
    try:
        return package_version("modlamp")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "The modlamp package is not installed."
        ) from error


def validate_modlamp_version(
    config: dict[str, Any],
    installed_version: str,
) -> None:
    """Verify the installed modlAMP version against the configuration."""
    modlamp_config = config["modlamp"]

    expected_version = str(
        modlamp_config.get("expected_version", "")
    ).strip()

    enforce_version = bool(
        modlamp_config.get("enforce_version", True)
    )

    if (
        enforce_version
        and expected_version
        and installed_version != expected_version
    ):
        raise ValueError(
            "Unexpected modlAMP version. "
            f"Expected {expected_version}, found {installed_version}."
        )


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate the required configuration fields."""
    required_sections = [
        "input",
        "dataset",
        "modlamp",
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

    dataset_config = config["dataset"]

    required_dataset_fields = [
        "task_column",
        "sequence_id_column",
        "sequence_column",
        "label_column",
        "length_column",
        "class_names",
    ]

    missing_dataset_fields = [
        field
        for field in required_dataset_fields
        if field not in dataset_config
    ]

    if missing_dataset_fields:
        raise ValueError(
            "Missing dataset configuration fields: "
            + ", ".join(missing_dataset_fields)
        )

    modlamp_config = config["modlamp"]

    batch_size = int(
        modlamp_config.get("batch_size", 5000)
    )

    if batch_size < 1:
        raise ValueError(
            "modlamp.batch_size must be greater than zero."
        )

    charge_config = modlamp_config.get("charge", {})

    if "ph" not in charge_config:
        raise ValueError(
            "modlamp.charge.ph must be configured."
        )

    hydrophobicity_config = modlamp_config.get(
        "hydrophobicity",
        {},
    )

    required_hydrophobicity_fields = [
        "scale",
        "window",
        "modality",
    ]

    missing_hydrophobicity_fields = [
        field
        for field in required_hydrophobicity_fields
        if field not in hydrophobicity_config
    ]

    if missing_hydrophobicity_fields:
        raise ValueError(
            "Missing modlamp.hydrophobicity fields: "
            + ", ".join(missing_hydrophobicity_fields)
        )

    modality = hydrophobicity_config["modality"]

    if modality not in {"max", "mean"}:
        raise ValueError(
            "modlamp.hydrophobicity.modality must be "
            "'max' or 'mean'."
        )

    charged_config = modlamp_config.get(
        "charged_residue_proportion",
        {},
    )

    if "scale" not in charged_config:
        raise ValueError(
            "modlamp.charged_residue_proportion.scale "
            "must be configured."
        )


def normalize_class_names(
    class_names: dict[Any, Any],
) -> dict[int, str]:
    """Convert YAML class-name keys to integer labels."""
    return {
        int(label): str(name)
        for label, name in class_names.items()
    }


def validate_input_columns(
    dataframe: pd.DataFrame,
    dataset_config: dict[str, Any],
) -> None:
    """Validate the columns required in the AMP task dataset."""
    required_columns = [
        dataset_config["task_column"],
        dataset_config["sequence_id_column"],
        dataset_config["sequence_column"],
        dataset_config["label_column"],
        dataset_config["length_column"],
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required input columns were not found: "
            + ", ".join(missing_columns)
        )


def validate_task_rows(
    dataframe: pd.DataFrame,
    task_column: str,
    sequence_id_column: str,
    label_column: str,
) -> None:
    """Validate task-specific uniqueness and class availability."""
    duplicate_mask = dataframe.duplicated(
        subset=[task_column, sequence_id_column],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            dataframe.loc[
                duplicate_mask,
                [task_column, sequence_id_column],
            ]
            .head(5)
            .astype(str)
            .agg(":".join, axis=1)
            .tolist()
        )

        raise ValueError(
            "Duplicated task/sequence combinations were found. "
            f"Examples: {', '.join(examples)}"
        )

    class_counts = dataframe.groupby(
        task_column,
        observed=True,
    )[label_column].nunique()

    incomplete_tasks = class_counts[class_counts < 2]

    if not incomplete_tasks.empty:
        raise ValueError(
            "The following tasks do not contain both classes: "
            + ", ".join(
                incomplete_tasks.index.astype(str)
            )
        )


def build_unique_sequence_table(
    dataframe: pd.DataFrame,
    sequence_id_column: str,
    sequence_column: str,
    length_column: str,
) -> pd.DataFrame:
    """Construct and validate one row per unique sequence identifier."""
    sequence_counts = dataframe.groupby(
        sequence_id_column,
        observed=True,
    )[sequence_column].nunique()

    inconsistent_ids = sequence_counts[
        sequence_counts > 1
    ]

    if not inconsistent_ids.empty:
        examples = ", ".join(
            inconsistent_ids.index.astype(str)[:5]
        )

        raise ValueError(
            "Some sequence identifiers map to multiple sequences. "
            f"Examples: {examples}"
        )

    sequence_table = (
        dataframe[
            [
                sequence_id_column,
                sequence_column,
                length_column,
            ]
        ]
        .drop_duplicates(
            subset=[sequence_id_column]
        )
        .copy()
    )

    sequence_table[sequence_id_column] = (
        sequence_table[sequence_id_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    sequence_table[sequence_column] = (
        sequence_table[sequence_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_id_mask = (
        sequence_table[sequence_id_column] == ""
    )

    if empty_id_mask.any():
        raise ValueError(
            "Empty sequence identifiers were found."
        )

    empty_sequence_mask = (
        sequence_table[sequence_column] == ""
    )

    if empty_sequence_mask.any():
        examples = (
            sequence_table.loc[
                empty_sequence_mask,
                sequence_id_column,
            ]
            .astype(str)
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Empty sequences were found. "
            f"Examples: {', '.join(examples)}"
        )

    lowercase_mask = (
        sequence_table[sequence_column]
        != sequence_table[sequence_column].str.upper()
    )

    if lowercase_mask.any():
        examples = (
            sequence_table.loc[
                lowercase_mask,
                sequence_id_column,
            ]
            .astype(str)
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Non-uppercase sequences were found. "
            f"Examples: {', '.join(examples)}"
        )

    sequence_table[length_column] = pd.to_numeric(
        sequence_table[length_column],
        errors="raise",
    ).astype(int)

    return sequence_table.sort_values(
        sequence_id_column
    ).reset_index(drop=True)


def descriptor_column(
    descriptor: np.ndarray,
    expected_rows: int,
    feature_name: str,
) -> np.ndarray:
    """Convert a one-dimensional modlAMP descriptor into a flat array."""
    values = np.asarray(
        descriptor,
        dtype=float,
    )

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if values.shape != (expected_rows, 1):
        raise ValueError(
            f"Unexpected shape for {feature_name}: "
            f"{values.shape}; expected ({expected_rows}, 1)."
        )

    return values[:, 0]


def scalar_scale_value(value: Any) -> float:
    """Convert a one-dimensional modlAMP scale value into a float."""
    values = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if values.size != 1:
        raise ValueError(
            "The configured charged-residue scale must be "
            "one-dimensional."
        )

    return float(values[0])


def derive_charged_residues(
    sequence: str,
    amino_acid_order: list[str],
    scale_name: str,
    threshold: float,
) -> list[str]:
    """
    Derive charged residues from a named modlAMP scale.

    Residues with an absolute scale value greater than the configured
    threshold are considered charged.
    """
    descriptor = PeptideDescriptor(
        sequence,
        scale_name,
    )

    charged_residues: list[str] = []

    for residue in amino_acid_order:
        if residue not in descriptor.scale:
            raise ValueError(
                f"Residue '{residue}' is missing from "
                f"modlAMP scale '{scale_name}'."
            )

        value = scalar_scale_value(
            descriptor.scale[residue]
        )

        if abs(value) > threshold:
            charged_residues.append(residue)

    if not charged_residues:
        raise ValueError(
            "No charged residues were identified from "
            f"modlAMP scale '{scale_name}'."
        )

    return charged_residues


def calculate_modlamp_batch(
    sequences: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Calculate modlAMP descriptors for one sequence batch."""
    modlamp_config = config["modlamp"]

    charge_config = modlamp_config["charge"]
    hydrophobicity_config = modlamp_config[
        "hydrophobicity"
    ]
    charged_config = modlamp_config[
        "charged_residue_proportion"
    ]
    composition_config = modlamp_config[
        "amino_acid_composition"
    ]

    number_of_sequences = len(sequences)

    if number_of_sequences == 0:
        raise ValueError(
            "Cannot calculate descriptors for an empty batch."
        )

    # Global descriptors
    global_descriptor = GlobalDescriptor(sequences)

    global_descriptor.length()
    sequence_lengths = descriptor_column(
        global_descriptor.descriptor,
        expected_rows=number_of_sequences,
        feature_name="sequence_length_modlamp",
    ).astype(int)

    global_descriptor.calculate_charge(
        ph=float(charge_config["ph"]),
        amide=bool(charge_config.get("amide", False)),
    )
    net_charge = descriptor_column(
        global_descriptor.descriptor,
        expected_rows=number_of_sequences,
        feature_name="net_charge",
    )

    global_descriptor.hydrophobic_ratio()
    hydrophobic_ratio = descriptor_column(
        global_descriptor.descriptor,
        expected_rows=number_of_sequences,
        feature_name="hydrophobic_ratio",
    )

    # Amino-acid composition from modlAMP
    composition_scale = composition_config.get(
        "scale",
        "relative",
    )

    global_descriptor.count_aa(
        scale=composition_scale
    )

    amino_acid_matrix = np.asarray(
        global_descriptor.descriptor,
        dtype=float,
    )

    amino_acid_order = [
        str(name)
        for name in global_descriptor.featurenames
    ]

    expected_composition_shape = (
        number_of_sequences,
        len(amino_acid_order),
    )

    if amino_acid_matrix.shape != expected_composition_shape:
        raise ValueError(
            "Unexpected amino-acid composition shape: "
            f"{amino_acid_matrix.shape}; expected "
            f"{expected_composition_shape}."
        )

    # Mean hydrophobicity from the configured modlAMP scale
    hydrophobicity_descriptor = PeptideDescriptor(
        sequences,
        hydrophobicity_config["scale"],
    )

    hydrophobicity_descriptor.calculate_global(
        window=int(hydrophobicity_config["window"]),
        modality=hydrophobicity_config["modality"],
    )

    mean_hydrophobicity = descriptor_column(
        hydrophobicity_descriptor.descriptor,
        expected_rows=number_of_sequences,
        feature_name="mean_hydrophobicity",
    )

    charged_residues = derive_charged_residues(
        sequence=sequences[0],
        amino_acid_order=amino_acid_order,
        scale_name=charged_config["scale"],
        threshold=float(
            charged_config.get("threshold", 0.0)
        ),
    )

    charged_indices = [
        amino_acid_order.index(residue)
        for residue in charged_residues
    ]

    charged_residue_proportion = (
        amino_acid_matrix[:, charged_indices].sum(
            axis=1
        )
    )

    properties: dict[str, Any] = {
        "sequence_length_modlamp": sequence_lengths,
        "net_charge": net_charge,
        "mean_hydrophobicity": mean_hydrophobicity,
        "charged_residue_proportion": (
            charged_residue_proportion
        ),
        "hydrophobic_ratio": hydrophobic_ratio,
    }

    for index, residue in enumerate(
        amino_acid_order
    ):
        properties[
            f"aa_fraction_{residue}"
        ] = amino_acid_matrix[:, index]

    return (
        pd.DataFrame(properties),
        amino_acid_order,
        charged_residues,
    )


def calculate_sequence_properties(
    sequence_table: pd.DataFrame,
    sequence_id_column: str,
    sequence_column: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Calculate descriptors in deterministic batches."""
    batch_size = int(
        config["modlamp"].get(
            "batch_size",
            5000,
        )
    )

    result_frames: list[pd.DataFrame] = []

    expected_amino_acid_order: list[str] | None = None
    expected_charged_residues: list[str] | None = None

    for start in range(
        0,
        len(sequence_table),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(sequence_table),
        )

        batch = sequence_table.iloc[
            start:stop
        ].copy()

        sequences = (
            batch[sequence_column]
            .astype(str)
            .tolist()
        )

        (
            batch_properties,
            amino_acid_order,
            charged_residues,
        ) = calculate_modlamp_batch(
            sequences=sequences,
            config=config,
        )

        if expected_amino_acid_order is None:
            expected_amino_acid_order = (
                amino_acid_order
            )
        elif amino_acid_order != expected_amino_acid_order:
            raise ValueError(
                "Amino-acid composition order changed "
                "between batches."
            )

        if expected_charged_residues is None:
            expected_charged_residues = (
                charged_residues
            )
        elif charged_residues != expected_charged_residues:
            raise ValueError(
                "Charged-residue definition changed "
                "between batches."
            )

        batch_properties.insert(
            0,
            sequence_column,
            batch[sequence_column].to_numpy(),
        )

        batch_properties.insert(
            0,
            sequence_id_column,
            batch[sequence_id_column].to_numpy(),
        )

        result_frames.append(
            batch_properties
        )

        print(
            f"Processed sequences {start + 1}-{stop} "
            f"of {len(sequence_table)}"
        )

    sequence_properties = pd.concat(
        result_frames,
        ignore_index=True,
    )

    return (
        sequence_properties,
        expected_amino_acid_order or [],
        expected_charged_residues or [],
    )


def validate_sequence_lengths(
    sequence_table: pd.DataFrame,
    sequence_properties: pd.DataFrame,
    sequence_id_column: str,
    length_column: str,
) -> None:
    """Compare input lengths with lengths calculated by modlAMP."""
    length_check = sequence_table[
        [
            sequence_id_column,
            length_column,
        ]
    ].merge(
        sequence_properties[
            [
                sequence_id_column,
                "sequence_length_modlamp",
            ]
        ],
        on=sequence_id_column,
        how="left",
        validate="one_to_one",
    )

    mismatch_mask = (
        length_check[length_column]
        != length_check["sequence_length_modlamp"]
    )

    if mismatch_mask.any():
        examples = (
            length_check.loc[
                mismatch_mask,
                sequence_id_column,
            ]
            .astype(str)
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Reported sequence lengths differ from "
            "modlAMP-calculated lengths. "
            f"Examples: {', '.join(examples)}"
        )


def describe_series(
    series: pd.Series,
) -> dict[str, float | int]:
    """Calculate descriptive statistics."""
    values = pd.to_numeric(
        series,
        errors="raise",
    ).dropna()

    if values.empty:
        raise ValueError(
            "Cannot summarize an empty numeric series."
        )

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))

    return {
        "n": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": (
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
        ),
        "median": float(values.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def build_class_distribution(
    dataframe: pd.DataFrame,
    task_column: str,
    label_column: str,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Build class counts and fractions for every task."""
    distribution = (
        dataframe.groupby(
            [task_column, label_column],
            observed=True,
        )
        .size()
        .reset_index(name="class_count")
    )

    task_totals = (
        dataframe.groupby(
            task_column,
            observed=True,
        )
        .size()
        .rename("total_count")
        .reset_index()
    )

    distribution = distribution.merge(
        task_totals,
        on=task_column,
        how="left",
        validate="many_to_one",
    )

    distribution["class_fraction"] = (
        distribution["class_count"]
        / distribution["total_count"]
    )

    distribution["class_name"] = (
        distribution[label_column]
        .map(class_names)
    )

    if distribution["class_name"].isna().any():
        raise ValueError(
            "At least one class label has no configured name."
        )

    return (
        distribution[
            [
                task_column,
                label_column,
                "class_name",
                "class_count",
                "total_count",
                "class_fraction",
            ]
        ]
        .sort_values(
            [task_column, label_column]
        )
        .reset_index(drop=True)
    )


def build_length_summary(
    dataframe: pd.DataFrame,
    task_column: str,
    label_column: str,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Summarize modlAMP-calculated lengths by task and class."""
    rows: list[dict[str, Any]] = []

    grouped = dataframe.groupby(
        [task_column, label_column],
        observed=True,
        sort=True,
    )

    for (task_id, label), group in grouped:
        rows.append(
            {
                "task_id": task_id,
                "amp_label": int(label),
                "class_name": class_names[int(label)],
                **describe_series(
                    group["sequence_length_modlamp"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_physicochemical_summary(
    dataframe: pd.DataFrame,
    task_column: str,
    label_column: str,
    class_names: dict[int, str],
    amino_acid_order: list[str],
) -> pd.DataFrame:
    """Summarize physicochemical and amino-acid composition features."""
    rows: list[dict[str, Any]] = []

    composition_features = [
        f"aa_fraction_{residue}"
        for residue in amino_acid_order
    ]

    grouped = dataframe.groupby(
        [task_column, label_column],
        observed=True,
        sort=True,
    )

    for (task_id, label), group in grouped:
        for feature in (
            PHYSICOCHEMICAL_FEATURES
            + composition_features
        ):
            feature_type = (
                "amino_acid_composition"
                if feature in composition_features
                else "physicochemical"
            )

            rows.append(
                {
                    "task_id": task_id,
                    "amp_label": int(label),
                    "class_name": class_names[int(label)],
                    "feature_type": feature_type,
                    "feature": feature,
                    **describe_series(group[feature]),
                }
            )

    return pd.DataFrame(rows)


def write_manifest(
    output_path: Path,
    *,
    input_path: Path,
    output_paths: dict[str, Path],
    dataframe: pd.DataFrame,
    sequence_properties: pd.DataFrame,
    task_column: str,
    modlamp_version: str,
    config: dict[str, Any],
    amino_acid_order: list[str],
    charged_residues: list[str],
) -> None:
    """Write a metadata record for the characterization execution."""
    manifest = {
        "input_file": str(input_path),
        "input_rows": int(len(dataframe)),
        "unique_sequences": int(
            len(sequence_properties)
        ),
        "tasks": sorted(
            dataframe[task_column]
            .astype(str)
            .unique()
            .tolist()
        ),
        "descriptor_engine": "modlAMP",
        "modlamp_version": modlamp_version,
        "modlamp_configuration": config["modlamp"],
        "amino_acid_order": amino_acid_order,
        "charged_residues_derived_from_scale": (
            charged_residues
        ),
        "outputs": {
            name: str(path)
            for name, path in output_paths.items()
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        json.dump(
            manifest,
            output_handle,
            indent=2,
            ensure_ascii=False,
        )
        output_handle.write("\n")


def check_output_paths(
    output_paths: list[Path],
    overwrite: bool,
) -> None:
    """Prevent accidental replacement of existing outputs."""
    if overwrite:
        return

    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_paths:
        raise FileExistsError(
            "Output files already exist: "
            + ", ".join(
                str(path)
                for path in existing_paths
            )
            + ". Use --overwrite to replace them."
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Characterize AMP classification tasks using "
            "modlAMP physicochemical descriptors."
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
        help="Path to dataset_characterization.yml.",
    )

    parser.add_argument(
        "--class-output",
        default=(
            "results/summaries/class_distribution.csv"
        ),
    )

    parser.add_argument(
        "--length-output",
        default=(
            "results/summaries/length_summary.csv"
        ),
    )

    parser.add_argument(
        "--physicochemical-output",
        default=(
            "results/summaries/"
            "physicochemical_summary.csv"
        ),
    )

    parser.add_argument(
        "--properties-output",
        default=(
            "results/summaries/"
            "sequence_properties.csv"
        ),
    )

    parser.add_argument(
        "--manifest-output",
        default=(
            "metadata/"
            "dataset_characterization_manifest.json"
        ),
    )

    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional input encoding override.",
    )

    parser.add_argument(
        "--separator",
        default=None,
        help=r"Optional separator override, such as ',' or '\t'.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing outputs to be replaced.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the modlAMP-based dataset characterization."""
    args = parse_arguments()

    input_path = Path(args.input)
    config_path = Path(args.config)

    class_output_path = Path(args.class_output)
    length_output_path = Path(args.length_output)
    physicochemical_output_path = Path(
        args.physicochemical_output
    )
    properties_output_path = Path(
        args.properties_output
    )
    manifest_output_path = Path(
        args.manifest_output
    )

    output_paths = {
        "class_distribution": class_output_path,
        "length_summary": length_output_path,
        "physicochemical_summary": (
            physicochemical_output_path
        ),
        "sequence_properties": properties_output_path,
        "manifest": manifest_output_path,
    }

    try:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        check_output_paths(
            output_paths=list(output_paths.values()),
            overwrite=args.overwrite,
        )

        config = load_config(config_path)
        validate_configuration(config)

        installed_modlamp_version = (
            get_modlamp_version()
        )

        validate_modlamp_version(
            config=config,
            installed_version=installed_modlamp_version,
        )

        input_config = config["input"]
        dataset_config = config["dataset"]

        encoding = (
            args.encoding
            or input_config.get(
                "encoding",
                "utf-8",
            )
        )

        separator_value = (
            args.separator
            if args.separator is not None
            else input_config.get(
                "separator",
                ",",
            )
        )

        separator = decode_separator(
            separator_value
        )

        dataframe = pd.read_csv(
            input_path,
            encoding=encoding,
            sep=separator,
        )

        validate_input_columns(
            dataframe=dataframe,
            dataset_config=dataset_config,
        )

        task_column = dataset_config["task_column"]
        sequence_id_column = dataset_config[
            "sequence_id_column"
        ]
        sequence_column = dataset_config[
            "sequence_column"
        ]
        label_column = dataset_config[
            "label_column"
        ]
        length_column = dataset_config[
            "length_column"
        ]

        included_tasks = dataset_config.get(
            "included_tasks"
        )

        if included_tasks:
            available_tasks = set(
                dataframe[task_column]
                .dropna()
                .astype(str)
                .unique()
            )

            missing_tasks = sorted(
                set(included_tasks)
                - available_tasks
            )

            if missing_tasks:
                raise ValueError(
                    "Configured tasks were not found: "
                    + ", ".join(missing_tasks)
                )

            dataframe = dataframe[
                dataframe[task_column].isin(
                    included_tasks
                )
            ].copy()

        if dataframe.empty:
            raise ValueError(
                "No rows remain after task filtering."
            )

        dataframe[label_column] = pd.to_numeric(
            dataframe[label_column],
            errors="raise",
        ).astype(int)

        class_names = normalize_class_names(
            dataset_config["class_names"]
        )

        unexpected_labels = sorted(
            set(dataframe[label_column].unique())
            - set(class_names)
        )

        if unexpected_labels:
            raise ValueError(
                "Unexpected class labels were found: "
                + ", ".join(
                    map(str, unexpected_labels)
                )
            )

        validate_task_rows(
            dataframe=dataframe,
            task_column=task_column,
            sequence_id_column=sequence_id_column,
            label_column=label_column,
        )

        sequence_table = build_unique_sequence_table(
            dataframe=dataframe,
            sequence_id_column=sequence_id_column,
            sequence_column=sequence_column,
            length_column=length_column,
        )

        (
            sequence_properties,
            amino_acid_order,
            charged_residues,
        ) = calculate_sequence_properties(
            sequence_table=sequence_table,
            sequence_id_column=sequence_id_column,
            sequence_column=sequence_column,
            config=config,
        )

        if dataset_config.get(
            "verify_reported_length",
            True,
        ):
            validate_sequence_lengths(
                sequence_table=sequence_table,
                sequence_properties=sequence_properties,
                sequence_id_column=sequence_id_column,
                length_column=length_column,
            )

        analysis_dataframe = dataframe.merge(
            sequence_properties,
            on=[
                sequence_id_column,
                sequence_column,
            ],
            how="left",
            validate="many_to_one",
        )

        if analysis_dataframe[
            PHYSICOCHEMICAL_FEATURES
        ].isna().any().any():
            raise ValueError(
                "Missing physicochemical values were found "
                "after merging descriptors."
            )

        class_distribution = (
            build_class_distribution(
                dataframe=analysis_dataframe,
                task_column=task_column,
                label_column=label_column,
                class_names=class_names,
            )
        )

        length_summary = build_length_summary(
            dataframe=analysis_dataframe,
            task_column=task_column,
            label_column=label_column,
            class_names=class_names,
        )

        physicochemical_summary = (
            build_physicochemical_summary(
                dataframe=analysis_dataframe,
                task_column=task_column,
                label_column=label_column,
                class_names=class_names,
                amino_acid_order=amino_acid_order,
            )
        )

        composition_features = [
            f"aa_fraction_{residue}"
            for residue in amino_acid_order
        ]

        sequence_output_columns = [
            task_column,
            sequence_id_column,
            sequence_column,
            label_column,
            length_column,
            "sequence_length_modlamp",
            *PHYSICOCHEMICAL_FEATURES,
            *composition_features,
        ]

        sequence_level_output = (
            analysis_dataframe[
                sequence_output_columns
            ]
            .sort_values(
                [
                    task_column,
                    label_column,
                    sequence_id_column,
                ]
            )
            .reset_index(drop=True)
        )

        for output_path in output_paths.values():
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        class_distribution.to_csv(
            class_output_path,
            index=False,
        )

        length_summary.to_csv(
            length_output_path,
            index=False,
        )

        physicochemical_summary.to_csv(
            physicochemical_output_path,
            index=False,
        )

        sequence_level_output.to_csv(
            properties_output_path,
            index=False,
        )

        write_manifest(
            output_path=manifest_output_path,
            input_path=input_path,
            output_paths=output_paths,
            dataframe=dataframe,
            sequence_properties=sequence_properties,
            task_column=task_column,
            modlamp_version=installed_modlamp_version,
            config=config,
            amino_acid_order=amino_acid_order,
            charged_residues=charged_residues,
        )

        print()
        print("Dataset characterization completed")
        print("----------------------------------")
        print(
            f"modlAMP version: "
            f"{installed_modlamp_version}"
        )
        print(
            f"Unique sequences processed: "
            f"{len(sequence_properties)}"
        )
        print(
            "Charged residues derived from "
            f"{config['modlamp']['charged_residue_proportion']['scale']}: "
            + ", ".join(charged_residues)
        )
        print()
        print(
            f"Class distribution: "
            f"{class_output_path}"
        )
        print(
            f"Length summary: "
            f"{length_output_path}"
        )
        print(
            f"Physicochemical summary: "
            f"{physicochemical_output_path}"
        )
        print(
            f"Sequence properties: "
            f"{properties_output_path}"
        )
        print(
            f"Execution manifest: "
            f"{manifest_output_path}"
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