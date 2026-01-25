from pathlib import Path

import pytest

import mcs


@pytest.mark.parametrize(
    "train_file,execution_file",
    [
        ("train_classification.yaml", "execution.yaml"),
        ("train_regression.yaml", "execution_gpu.yaml"),
    ],
)
def test_run_pack_creates_registry_records(tmp_path: Path, train_file: str, execution_file: str):
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"

    if not examples_dir.exists():
        pytest.skip("Missing examples/ directory at repo root.")

    out = mcs.run_pack(
        examples_dir,
        train_file=train_file,
        execution_file=execution_file,
        registry_root=tmp_path / ".mcs_registry",
        strict=False,                  # allow INFO/WARNING
        artifact_mode="reference",     # no real files required
        overwrite=True,
        notes="pytest run",
    )

    record = out["record"]
    registry_paths = out["registry_paths"]

    # 1) run record exists
    run_json = Path(registry_paths["run"])
    assert run_json.exists(), "runs/<run_id>.json should exist"
    assert record.run_id in run_json.name

    # 2) specs exist at their fingerprint paths
    specs = registry_paths["specs"]
    for k in ["dataset", "split", "embedding", "train", "execution"]:
        p = Path(specs[k])
        assert p.exists(), f"spec registry file missing for {k}: {p}"

    # 3) registry layout sanity
    reg_root = tmp_path / ".mcs_registry"
    assert (reg_root / "runs").exists()
    assert (reg_root / "specs").exists()
    assert (reg_root / "artifacts").exists()
