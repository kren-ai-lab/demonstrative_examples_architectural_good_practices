#!/usr/bin/env python3

"""Generate one frozen ESM2 representation per unique sequence using Sylphy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Calculate the SHA-256 checksum of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def decode_separator(value: str) -> str:
    """Convert an escaped separator such as '\\t' into one character."""
    decoded = bytes(value, "utf-8").decode("unicode_escape")

    if len(decoded) != 1:
        raise ValueError(
            "The separator must resolve to exactly one character."
        )

    return decoded


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def package_version(name: str) -> str:
    """Return an installed package version."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate the principal configuration fields."""
    for section in ("input", "dataset", "representation", "output"):
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    dataset = config["dataset"]

    for field in (
        "task_column",
        "sequence_id_column",
        "sequence_column",
        "alphabet",
    ):
        if field not in dataset:
            raise ValueError(
                f"Missing dataset configuration field: {field}"
            )

    representation = config["representation"]

    if representation.get("framework") != "sylphy":
        raise ValueError(
            "representation.framework must be 'sylphy'."
        )

    for subsection in ("sylphy", "model", "extraction"):
        if subsection not in representation:
            raise ValueError(
                f"Missing representation subsection: {subsection}"
            )

    model = representation["model"]
    extraction = representation["extraction"]

    if not model.get("repository"):
        raise ValueError(
            "representation.model.repository must be configured."
        )

    if "esm2" not in str(model.get("registry_name", "")).lower():
        raise ValueError(
            "representation.model.registry_name must contain 'esm2' "
            "so Sylphy selects its ESM2 backend."
        )

    if extraction.get("pooling") != "mean":
        raise ValueError(
            "This implementation currently supports mean pooling only."
        )

    if extraction.get("layer_aggregation") != "mean":
        raise ValueError(
            "This implementation currently supports mean layer "
            "aggregation only."
        )

    if extraction.get("precision") != "fp32":
        raise ValueError(
            "This experiment is configured for fp32 extraction."
        )

    if int(extraction.get("batch_size", 0)) < 1:
        raise ValueError(
            "representation.extraction.batch_size must be positive."
        )

    if int(extraction.get("max_length", 0)) < 1:
        raise ValueError(
            "representation.extraction.max_length, interpreted as the maximum "
            "number of residues, must be positive."
        )


def check_outputs(paths: list[Path], overwrite: bool) -> None:
    """Prevent accidental replacement of existing outputs."""
    if overwrite:
        return

    existing = [path for path in paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite to replace them."
        )


def load_unique_sequences(
    input_path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load both tasks and construct one deterministic row per sequence."""
    input_config = config["input"]
    dataset = config["dataset"]

    task_column = dataset["task_column"]
    sequence_id_column = dataset["sequence_id_column"]
    sequence_column = dataset["sequence_column"]

    dataframe = pd.read_csv(
        input_path,
        sep=decode_separator(str(input_config.get("separator", ","))),
        encoding=str(input_config.get("encoding", "utf-8")),
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
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Required input columns were not found: "
            + ", ".join(missing)
        )

    for column in required_columns:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
        )

    included_tasks = dataset.get("included_tasks")

    if included_tasks:
        available_tasks = set(
            dataframe[task_column].unique()
        )
        missing_tasks = sorted(
            set(included_tasks) - available_tasks
        )

        if missing_tasks:
            raise ValueError(
                "Configured tasks were not found: "
                + ", ".join(missing_tasks)
            )

        dataframe = dataframe[
            dataframe[task_column].isin(included_tasks)
        ].copy()

    if dataframe.empty:
        raise ValueError(
            "No rows remain after task filtering."
        )

    if (dataframe[sequence_id_column] == "").any():
        raise ValueError(
            "Empty sequence identifiers were found."
        )

    if (dataframe[sequence_column] == "").any():
        raise ValueError(
            "Empty sequences were found."
        )

    id_to_sequence_counts = dataframe.groupby(
        sequence_id_column,
        observed=True,
    )[sequence_column].nunique()

    inconsistent_ids = id_to_sequence_counts[
        id_to_sequence_counts > 1
    ]

    if not inconsistent_ids.empty:
        raise ValueError(
            "Some sequence identifiers map to multiple sequences. "
            "Examples: "
            + ", ".join(
                inconsistent_ids.index.astype(str)[:5]
            )
        )

    sequence_to_id_counts = dataframe.groupby(
        sequence_column,
        observed=True,
    )[sequence_id_column].nunique()

    inconsistent_sequences = sequence_to_id_counts[
        sequence_to_id_counts > 1
    ]

    if not inconsistent_sequences.empty:
        raise ValueError(
            "Some normalized sequences map to multiple sequence IDs."
        )

    alphabet = set(str(dataset["alphabet"]))
    invalid_examples: list[str] = []

    unique_sequences = (
        dataframe[
            [
                sequence_id_column,
                sequence_column,
            ]
        ]
        .drop_duplicates(
            subset=[sequence_id_column]
        )
        .sort_values(sequence_id_column)
        .reset_index(drop=True)
    )

    for sequence_id, sequence in unique_sequences.itertuples(
        index=False,
        name=None,
    ):
        invalid_characters = sorted(
            set(sequence) - alphabet
        )

        if invalid_characters:
            invalid_examples.append(
                f"{sequence_id}:{''.join(invalid_characters)}"
            )

            if len(invalid_examples) == 5:
                break

    if invalid_examples:
        raise ValueError(
            "Sequences contain characters outside the configured "
            "alphabet: "
            + "; ".join(invalid_examples)
        )

    if dataset.get("verify_sequence_id_sha256", True):
        mismatched_ids = [
            sequence_id
            for sequence_id, sequence
            in unique_sequences.itertuples(
                index=False,
                name=None,
            )
            if sha256_text(sequence) != sequence_id
        ]

        if mismatched_ids:
            raise ValueError(
                "sequence_id does not match SHA-256(sequence) for "
                f"{len(mismatched_ids)} sequences. Examples: "
                + ", ".join(mismatched_ids[:5])
            )

    canonical_text = "".join(
        f"{sequence_id}\t{sequence}\n"
        for sequence_id, sequence
        in unique_sequences.itertuples(
            index=False,
            name=None,
        )
    )

    task_counts = (
        dataframe.groupby(
            task_column,
            observed=True,
        )
        .size()
        .astype(int)
        .to_dict()
    )

    summary = {
        "input_rows": int(len(dataframe)),
        "unique_sequences": int(len(unique_sequences)),
        "tasks": sorted(
            dataframe[task_column].unique().tolist()
        ),
        "task_rows": {
            str(key): int(value)
            for key, value in task_counts.items()
        },
        "canonical_sequence_table_sha256": sha256_text(
            canonical_text
        ),
        "minimum_sequence_length": int(
            unique_sequences[sequence_column].str.len().min()
        ),
        "maximum_sequence_length": int(
            unique_sequences[sequence_column].str.len().max()
        ),
    }

    return unique_sequences, summary


def resolve_model_revision(
    repository: str,
    requested_revision: str | None,
    resolve_to_commit: bool,
) -> str | None:
    """Resolve a Hugging Face revision to an immutable commit SHA."""
    if not resolve_to_commit:
        return requested_revision

    try:
        from huggingface_hub import model_info
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to resolve the model revision."
        ) from error

    info = model_info(
        repo_id=repository,
        revision=requested_revision or "main",
    )

    if not info.sha:
        raise RuntimeError(
            "Hugging Face did not return a model commit SHA."
        )

    return str(info.sha)


def resolve_device(
    requested_device: str,
    strict_device: bool,
) -> str:
    """Resolve the requested execution device."""
    import torch

    requested = requested_device.lower()

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested not in {"cuda", "cpu"}:
        raise ValueError(
            "Device must be 'cuda', 'cpu', or 'auto'."
        )

    if requested == "cuda" and not torch.cuda.is_available():
        if strict_device:
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return "cpu"

    return requested


def set_reproducibility(
    seed: int,
    allow_tf32: bool,
) -> None:
    """Set deterministic inference-related runtime controls."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_layer_spec(
    layer_spec: Any,
    number_of_hidden_states: int,
) -> list[int]:
    """Resolve a layer specification against returned hidden states."""
    if isinstance(layer_spec, int):
        indices = [layer_spec]
    elif isinstance(layer_spec, str):
        normalized = layer_spec.strip().lower()

        if normalized == "last":
            indices = [number_of_hidden_states - 1]
        elif normalized == "last4":
            indices = list(
                range(
                    max(0, number_of_hidden_states - 4),
                    number_of_hidden_states,
                )
            )
        else:
            try:
                indices = [int(normalized)]
            except ValueError as error:
                raise ValueError(
                    f"Unsupported layer specification: {layer_spec!r}"
                ) from error
    elif isinstance(layer_spec, list):
        indices = [int(value) for value in layer_spec]
    else:
        raise ValueError(
            f"Unsupported layer specification: {layer_spec!r}"
        )

    resolved: list[int] = []

    for index in indices:
        actual_index = (
            index
            if index >= 0
            else number_of_hidden_states + index
        )

        if not 0 <= actual_index < number_of_hidden_states:
            raise ValueError(
                f"Layer index {index} is outside the available "
                f"range for {number_of_hidden_states} hidden states."
            )

        resolved.append(actual_index)

    return sorted(set(resolved))


def aggregate_layers(
    hidden_states: tuple[Any, ...],
    selected_indices: list[int],
    aggregation: str,
) -> Any:
    """Aggregate selected hidden states."""
    import torch

    selected = [
        hidden_states[index]
        for index in selected_indices
    ]

    if len(selected) == 1:
        return selected[0]

    if aggregation == "mean":
        return torch.stack(selected, dim=0).mean(dim=0)

    raise ValueError(
        f"Unsupported layer aggregation: {aggregation}"
    )


def get_special_tokens_mask(
    tokenizer: Any,
    input_ids: Any,
    provided_mask: Any | None,
) -> Any:
    """Return a tensor marking special tokens and padding."""
    import torch

    if provided_mask is not None:
        return provided_mask.bool()

    masks = [
        tokenizer.get_special_tokens_mask(
            row.tolist(),
            already_has_special_tokens=True,
        )
        for row in input_ids.detach().cpu()
    ]

    return torch.tensor(
        masks,
        dtype=torch.bool,
        device=input_ids.device,
    )


def verify_no_truncation(
    sequences: list[str],
    maximum_residues: int,
) -> None:
    """Fail when a raw sequence exceeds the configured residue limit."""
    too_long = [
        (index, len(sequence))
        for index, sequence in enumerate(sequences)
        if len(sequence) > maximum_residues
    ]

    if too_long:
        examples = ", ".join(
            f"batch_index={index},residues={length}"
            for index, length in too_long[:5]
        )

        raise ValueError(
            f"{len(too_long)} sequences exceed the configured "
            f"maximum of {maximum_residues} residues. "
            f"Examples: {examples}"
        )


def extract_embeddings(
    embedder: Any,
    sequence_table: pd.DataFrame,
    sequence_column: str,
    extraction: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract special-token-aware mean-pooled embeddings."""
    import torch

    embedder.ensure_loaded()

    tokenizer = embedder.tokenizer
    model = embedder.model

    if tokenizer is None or model is None:
        raise RuntimeError(
            "Sylphy did not expose a loaded tokenizer and model."
        )

    model.eval()

    sequences = sequence_table[sequence_column].tolist()
    maximum_residues = int(extraction["max_length"])
    special_tokens_per_sequence = int(
        tokenizer.num_special_tokens_to_add(pair=False)
    )
    tokenizer_max_length = (
        maximum_residues + special_tokens_per_sequence
    )

    model_max_positions = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        model_max_positions is not None
        and tokenizer_max_length > int(model_max_positions)
    ):
        raise ValueError(
            "The requested residue limit requires "
            f"{tokenizer_max_length} model tokens, but the model "
            f"supports only {model_max_positions} positions."
        )
    requested_batch_size = int(extraction["batch_size"])
    batch_size = requested_batch_size
    oom_backoff = bool(extraction.get("oom_backoff", True))
    remove_special_tokens = bool(
        extraction.get("remove_special_tokens", True)
    )
    layer_spec = extraction.get("layer", "last")
    layer_aggregation = str(
        extraction.get("layer_aggregation", "mean")
    )

    if extraction.get("fail_on_truncation", True):
        for start in range(0, len(sequences), requested_batch_size):
            verify_no_truncation(
                sequences[start:start + requested_batch_size],
                maximum_residues,
            )

    matrices: list[np.ndarray] = []
    observed_batch_sizes: list[int] = []
    selected_layer_indices: list[int] | None = None
    index = 0

    while index < len(sequences):
        current_sequences = sequences[
            index:index + batch_size
        ]

        try:
            encoded = tokenizer(
                current_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )

            special_tokens_mask = encoded.pop(
                "special_tokens_mask",
                None,
            )

            model_inputs = {
                key: value.to(embedder.device)
                for key, value in encoded.items()
            }

            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    output_hidden_states=True,
                )

            hidden_states = outputs.hidden_states

            if hidden_states is None:
                hidden_states = (
                    outputs.last_hidden_state,
                )

            current_layer_indices = parse_layer_spec(
                layer_spec,
                len(hidden_states),
            )

            if selected_layer_indices is None:
                selected_layer_indices = (
                    current_layer_indices
                )
            elif (
                current_layer_indices
                != selected_layer_indices
            ):
                raise RuntimeError(
                    "Resolved layer indices changed between batches."
                )

            representations = aggregate_layers(
                hidden_states,
                current_layer_indices,
                layer_aggregation,
            )

            attention_mask = model_inputs[
                "attention_mask"
            ].bool()

            if remove_special_tokens:
                special_mask = get_special_tokens_mask(
                    tokenizer,
                    model_inputs["input_ids"],
                    (
                        special_tokens_mask.to(
                            embedder.device
                        )
                        if special_tokens_mask is not None
                        else None
                    ),
                )

                pooling_mask = (
                    attention_mask
                    & ~special_mask
                )
            else:
                pooling_mask = attention_mask

            residue_counts = pooling_mask.sum(dim=1)

            if (residue_counts == 0).any():
                raise ValueError(
                    "At least one sequence has no residue tokens "
                    "available for mean pooling."
                )

            float_mask = pooling_mask.unsqueeze(-1).to(
                representations.dtype
            )

            pooled = (
                representations * float_mask
            ).sum(dim=1) / float_mask.sum(
                dim=1
            ).clamp_min(1.0)

            matrix = (
                pooled.detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            matrices.append(matrix)
            observed_batch_sizes.append(
                len(current_sequences)
            )
            index += len(current_sequences)

        except torch.cuda.OutOfMemoryError:
            if (
                not oom_backoff
                or batch_size <= 1
                or embedder.device.type != "cuda"
            ):
                raise

            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()

    matrix = np.vstack(matrices)

    runtime_summary = {
        "requested_batch_size": requested_batch_size,
        "minimum_effective_batch_size": int(
            min(observed_batch_sizes)
        ),
        "maximum_effective_batch_size": int(
            max(observed_batch_sizes)
        ),
        "selected_hidden_state_indices": (
            selected_layer_indices or []
        ),
        "maximum_sequence_length_residues": maximum_residues,
        "special_tokens_per_sequence": special_tokens_per_sequence,
        "tokenizer_max_length_tokens": tokenizer_max_length,
        "number_of_batches": int(
            len(observed_batch_sizes)
        ),
    }

    return matrix, runtime_summary


def validate_embeddings(
    sequence_table: pd.DataFrame,
    embeddings: np.ndarray,
    expected_dimension: int,
) -> None:
    """Validate the final embedding matrix."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be two-dimensional, found {embeddings.shape}."
        )

    if embeddings.shape[0] != len(sequence_table):
        raise ValueError(
            "The number of embeddings does not match the number "
            "of unique sequences."
        )

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected embedding dimension {expected_dimension}, "
            f"found {embeddings.shape[1]}."
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, found {embeddings.dtype}."
        )

    if np.isnan(embeddings).any():
        raise ValueError(
            "NaN values were found in the embeddings."
        )

    if np.isinf(embeddings).any():
        raise ValueError(
            "Infinite values were found in the embeddings."
        )


def hardware_summary(device: str) -> dict[str, Any]:
    """Collect execution hardware metadata."""
    import torch

    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "requested_runtime_device": device,
        "torch_device": device,
        "cpu_count": os.cpu_count(),
    }

    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)

        summary["cuda"] = {
            "device_index": int(index),
            "device_name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(
                properties.total_memory
            ),
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
            ),
        }

    return summary


def write_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    """Write deterministic JSON metadata."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single frozen ESM2-T6 representation for "
            "the union of sequences across both AMP tasks."
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
        help="Path to configs/representation.yml.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional embeddings output override.",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional representation manifest override.",
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Optional execution-device override.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing outputs to be replaced.",
    )

    return parser.parse_args()


def main() -> int:
    """Generate and validate the frozen representation asset."""
    args = parse_arguments()

    input_path = Path(args.input)
    config_path = Path(args.config)

    try:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        config = load_yaml(config_path)
        validate_configuration(config)

        output_config = config["output"]
        representation = config["representation"]
        sylphy_config = representation["sylphy"]
        model_config = representation["model"]
        extraction = representation["extraction"].copy()

        if args.device is not None:
            extraction["device"] = args.device

        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError(
                    "--batch-size must be positive."
                )

            extraction["batch_size"] = args.batch_size

        output_path = Path(
            args.output or output_config["embeddings"]
        )
        manifest_path = Path(
            args.manifest or output_config["manifest"]
        )

        check_outputs(
            [output_path, manifest_path],
            overwrite=args.overwrite,
        )

        installed_sylphy_version = package_version(
            "sylphy"
        )
        expected_sylphy_version = str(
            sylphy_config.get(
                "expected_version",
                "",
            )
        )

        if (
            sylphy_config.get("enforce_version", True)
            and installed_sylphy_version
            != expected_sylphy_version
        ):
            raise RuntimeError(
                "Unexpected Sylphy version. "
                f"Expected {expected_sylphy_version}, "
                f"found {installed_sylphy_version}."
            )

        cache_root = sylphy_config.get("cache_root")

        if cache_root:
            os.environ["SYLPHY_CACHE_ROOT"] = str(
                Path(cache_root).expanduser().resolve()
            )

        sequence_table, input_summary = (
            load_unique_sequences(
                input_path=input_path,
                config=config,
            )
        )

        repository = str(
            model_config["repository"]
        )
        requested_revision = model_config.get(
            "requested_revision"
        )
        resolved_revision = resolve_model_revision(
            repository=repository,
            requested_revision=(
                str(requested_revision)
                if requested_revision is not None
                else None
            ),
            resolve_to_commit=bool(
                model_config.get(
                    "resolve_revision_to_commit",
                    True,
                )
            ),
        )

        from sylphy.core.model_registry import register_model
        from sylphy.core.model_spec import ModelSpec
        from sylphy.embedding_extractor import create_embedding

        registry_name = str(
            model_config["registry_name"]
        )

        register_model(
            ModelSpec(
                name=registry_name,
                provider="huggingface",
                ref=repository,
                revision=resolved_revision,
            )
        )

        actual_device = resolve_device(
            requested_device=str(
                extraction.get("device", "cuda")
            ),
            strict_device=bool(
                extraction.get(
                    "strict_device",
                    True,
                )
            ),
        )

        seed = int(
            extraction.get("seed", 42)
        )

        set_reproducibility(
            seed=seed,
            allow_tf32=bool(
                extraction.get(
                    "allow_tf32",
                    False,
                )
            ),
        )

        dataset_config = config["dataset"]
        sequence_id_column = dataset_config[
            "sequence_id_column"
        ]
        sequence_column = dataset_config[
            "sequence_column"
        ]

        sylphy_input = sequence_table[
            [sequence_column]
        ].copy()

        embedder = create_embedding(
            model_name=registry_name,
            dataset=sylphy_input,
            column_seq=sequence_column,
            name_device=actual_device,
            precision="fp32",
            oom_backoff=bool(
                extraction.get(
                    "oom_backoff",
                    True,
                )
            ),
        )

        started_at = utc_now()
        started = time.monotonic()

        embeddings, runtime_summary = (
            extract_embeddings(
                embedder=embedder,
                sequence_table=sequence_table,
                sequence_column=sequence_column,
                extraction=extraction,
            )
        )

        elapsed_seconds = time.monotonic() - started
        finished_at = utc_now()

        model = embedder.model

        # Sylphy may release resources only through run_process(). Our
        # custom pooling path explicitly releases them after metadata capture.
        model_metadata = {
            "model_class": (
                type(model).__name__
                if model is not None
                else None
            ),
            "tokenizer_class": (
                type(embedder.tokenizer).__name__
                if embedder.tokenizer is not None
                else None
            ),
            "hidden_size": (
                int(model.config.hidden_size)
                if model is not None
                else None
            ),
            "num_hidden_layers": (
                int(model.config.num_hidden_layers)
                if model is not None
                else None
            ),
            "vocab_size": (
                int(model.config.vocab_size)
                if model is not None
                else None
            ),
        }

        expected_dimension = int(
            model_config["expected_hidden_size"]
        )
        expected_layers = int(
            model_config["expected_num_hidden_layers"]
        )

        if (
            model_metadata["hidden_size"]
            != expected_dimension
        ):
            raise ValueError(
                "Loaded model hidden size does not match the "
                "configured expectation."
            )

        if (
            model_metadata["num_hidden_layers"]
            != expected_layers
        ):
            raise ValueError(
                "Loaded model layer count does not match the "
                "configured expectation."
            )

        validate_embeddings(
            sequence_table=sequence_table,
            embeddings=embeddings,
            expected_dimension=expected_dimension,
        )

        prefix = str(
            output_config.get(
                "embedding_column_prefix",
                "embedding_",
            )
        )
        width = max(
            3,
            len(str(expected_dimension - 1)),
        )
        embedding_columns = [
            f"{prefix}{index:0{width}d}"
            for index in range(expected_dimension)
        ]

        output_dataframe = pd.concat(
            [
                sequence_table[
                    [
                        sequence_id_column,
                        sequence_column,
                    ]
                ].reset_index(drop=True),
                pd.DataFrame(
                    embeddings,
                    columns=embedding_columns,
                ),
            ],
            axis=1,
        )

        if not output_dataframe[
            sequence_id_column
        ].is_monotonic_increasing:
            raise RuntimeError(
                "Output sequence identifiers are not sorted."
            )

        if output_dataframe[
            sequence_id_column
        ].duplicated().any():
            raise RuntimeError(
                "Duplicated sequence identifiers were found "
                "before export."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_dataframe.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression=str(
                output_config.get(
                    "compression",
                    "zstd",
                )
            ),
        )

        readback = pd.read_parquet(
            output_path,
            engine="pyarrow",
        )

        expected_ids = output_dataframe[
            sequence_id_column
        ].tolist()
        observed_ids = readback[
            sequence_id_column
        ].tolist()

        if expected_ids != observed_ids:
            raise RuntimeError(
                "Parquet readback changed sequence order or IDs."
            )

        if readback.shape != output_dataframe.shape:
            raise RuntimeError(
                "Parquet readback shape differs from the "
                "in-memory output."
            )

        readback_matrix = readback[
            embedding_columns
        ].to_numpy(dtype=np.float32)

        if not np.array_equal(
            embeddings,
            readback_matrix,
        ):
            raise RuntimeError(
                "Parquet readback changed embedding values."
            )

        hardware = hardware_summary(
            actual_device
        )

        manifest = {
            "schema_version": "1.0",
            "created_at_utc": utc_now(),
            "purpose": (
                "Frozen sequence-level representation shared by all "
                "tasks, similarity regimes, and partition seeds."
            ),
            "input": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                **input_summary,
            },
            "representation": {
                "framework": "sylphy",
                "model_repository": repository,
                "registry_name": registry_name,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "model_metadata": model_metadata,
                "layer": extraction.get("layer", "last"),
                "layer_aggregation": extraction.get(
                    "layer_aggregation",
                    "mean",
                ),
                "resolved_hidden_state_indices": (
                    runtime_summary[
                        "selected_hidden_state_indices"
                    ]
                ),
                "pooling": extraction["pooling"],
                "remove_special_tokens": bool(
                    extraction[
                        "remove_special_tokens"
                    ]
                ),
                "precision": extraction["precision"],
                "output_dimension": int(
                    embeddings.shape[1]
                ),
                "maximum_sequence_length_residues": int(
                    extraction["max_length"]
                ),
                "special_tokens_per_sequence": int(
                    runtime_summary[
                        "special_tokens_per_sequence"
                    ]
                ),
                "tokenizer_max_length_tokens": int(
                    runtime_summary[
                        "tokenizer_max_length_tokens"
                    ]
                ),
                "fail_on_truncation": bool(
                    extraction.get(
                        "fail_on_truncation",
                        True,
                    )
                ),
                "seed": seed,
                "allow_tf32": bool(
                    extraction.get(
                        "allow_tf32",
                        False,
                    )
                ),
            },
            "execution": {
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "elapsed_seconds": elapsed_seconds,
                "device": actual_device,
                **runtime_summary,
                "hardware": hardware,
            },
            "software": {
                "python": sys.version,
                "sylphy": installed_sylphy_version,
                "torch": package_version("torch"),
                "transformers": package_version(
                    "transformers"
                ),
                "huggingface_hub": package_version(
                    "huggingface-hub"
                ),
                "pandas": package_version("pandas"),
                "numpy": package_version("numpy"),
                "pyarrow": package_version("pyarrow"),
                "pyyaml": package_version("PyYAML"),
            },
            "output": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "size_bytes": int(
                    output_path.stat().st_size
                ),
                "rows": int(
                    output_dataframe.shape[0]
                ),
                "columns": int(
                    output_dataframe.shape[1]
                ),
                "embedding_columns": int(
                    len(embedding_columns)
                ),
                "compression": str(
                    output_config.get(
                        "compression",
                        "zstd",
                    )
                ),
            },
            "validations": {
                "one_representation_per_sequence_id": True,
                "constant_dimensionality": True,
                "no_nan": True,
                "no_infinite_values": True,
                "sorted_by_sequence_id": True,
                "input_output_identifier_correspondence": True,
                "parquet_readback_exact": True,
                "sequence_id_matches_sequence_sha256": bool(
                    dataset_config.get(
                        "verify_sequence_id_sha256",
                        True,
                    )
                ),
            },
        }

        write_manifest(
            path=manifest_path,
            manifest=manifest,
        )

        embedder.release_resources()

        print("Representation generation completed")
        print("-----------------------------------")
        print(f"Unique sequences: {len(sequence_table)}")
        print(
            f"Model: {repository}@{resolved_revision}"
        )
        print(
            "Embedding shape: "
            f"{embeddings.shape[0]} x {embeddings.shape[1]}"
        )
        print(f"Device: {actual_device}")
        print(
            f"Elapsed seconds: {elapsed_seconds:.3f}"
        )
        print(f"Embeddings: {output_path}")
        print(f"Manifest: {manifest_path}")
        print(
            f"Output SHA-256: {manifest['output']['sha256']}"
        )

        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        ImportError,
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
