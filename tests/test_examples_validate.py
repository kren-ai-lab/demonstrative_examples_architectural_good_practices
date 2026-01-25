from pathlib import Path

import pytest
import yaml

from mcs.schemas import DatasetSpec, SplitSpec, EmbeddingSpec, TrainSpec, ExecutionSpec
from mcs.validation import validate_all_5specs


def _load_yaml(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at '{path}' must contain a mapping (dict-like).")
    return data


@pytest.mark.parametrize(
    "train_file,execution_file",
    [
        ("train_classification.yaml", "execution.yaml"),
        ("train_classification.yaml", "execution_gpu.yaml"),
        ("train_regression.yaml", "execution.yaml"),
        ("train_regression.yaml", "execution_gpu.yaml"),
    ],
)
def test_examples_validate_5specs(train_file: str, execution_file: str):
    repo_root = Path(__file__).resolve().parents[1]
    ex = repo_root / "examples"

    # Required YAMLs
    dataset_yaml = ex / "dataset.yaml"
    split_yaml = ex / "split.yaml"
    embedding_yaml = ex / "embedding.yaml"
    train_yaml = ex / train_file
    execution_yaml = ex / execution_file

    missing = [p for p in [dataset_yaml, split_yaml, embedding_yaml, train_yaml, execution_yaml] if not p.exists()]
    if missing:
        pytest.skip(f"Missing example files: {', '.join(str(m) for m in missing)}")

    dataset = DatasetSpec.model_validate(_load_yaml(dataset_yaml))
    split = SplitSpec.model_validate(_load_yaml(split_yaml))
    embedding = EmbeddingSpec.model_validate(_load_yaml(embedding_yaml))
    train = TrainSpec.model_validate(_load_yaml(train_yaml))
    execution = ExecutionSpec.model_validate(_load_yaml(execution_yaml))

    # strict=True must not raise (ERROR issues are not allowed)
    issues = validate_all_5specs(dataset, split, embedding, train, execution, strict=True)
    assert isinstance(issues, list)
