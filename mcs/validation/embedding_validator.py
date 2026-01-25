from __future__ import annotations

from typing import List

from mcs.schemas.embedding import EmbeddingSpec
from mcs.validation.types import Severity, ValidationIssue


def _issue(
    severity: Severity,
    code: str,
    message: str,
    path: str | None = None,
    hint: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def validate_embedding_spec(spec: EmbeddingSpec) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Provider/model_id hygiene
    if not spec.model_id.strip():
        issues.append(_issue("ERROR", "EMB_MODEL_ID_EMPTY", "model_id must be non-empty.", path="model_id"))

    # Tokenization
    tok = spec.tokenization
    if tok.max_length <= 0:
        issues.append(_issue("ERROR", "EMB_MAXLEN_INVALID", "tokenization.max_length must be > 0.", path="tokenization.max_length"))
    if tok.padding == "max_length" and not tok.truncation:
        issues.append(
            _issue(
                "ERROR",
                "EMB_PAD_MAXLEN_NO_TRUNC",
                "padding='max_length' requires truncation=True to ensure bounded length.",
                path="tokenization",
            )
        )

    # Layer/pooling decisions
    if spec.pooling == "cls" and tok.add_special_tokens is False:
        issues.append(
            _issue(
                "WARNING",
                "EMB_CLS_NO_SPECIAL_TOKENS",
                "pooling='cls' but add_special_tokens=False; CLS token may be missing depending on model.",
                path="tokenization.add_special_tokens",
            )
        )

    # Compute hints
    if spec.device == "cpu" and spec.batch_size >= 64:
        issues.append(
            _issue(
                "INFO",
                "EMB_CPU_LARGE_BATCH",
                f"batch_size={spec.batch_size} on CPU may be slow / memory heavy.",
                path="batch_size",
            )
        )

    # Output
    if spec.output is None:
        issues.append(
            _issue(
                "INFO",
                "EMB_NO_OUTPUT",
                "EmbeddingSpec has no output declared; acceptable but reduces traceability.",
                path="output",
                hint="Consider declaring output.path and output.format for registry integration.",
            )
        )
    else:
        if not spec.output.path.strip():
            issues.append(_issue("ERROR", "EMB_OUTPUT_EMPTY_PATH", "output.path cannot be empty.", path="output.path"))
        if spec.output.checksum is None:
            issues.append(
                _issue(
                    "INFO",
                    "EMB_OUTPUT_NO_CHECKSUM",
                    "Embedding output has no checksum declared.",
                    path="output.checksum",
                )
            )

    # Dataset linkage
    if spec.input_dataset_fingerprint is None:
        issues.append(
            _issue(
                "INFO",
                "EMB_NO_DATASET_LINK",
                "input_dataset_fingerprint not set (optional).",
                path="input_dataset_fingerprint",
                hint="You can fill it after computing DatasetSpec.fingerprint().",
            )
        )

    return issues
