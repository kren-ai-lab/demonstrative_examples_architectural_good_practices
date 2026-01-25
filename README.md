# **Minimal Community Standard (MCS)**

*A lightweight, verifiable standard for reproducible machine learning workflows in protein science*

---

## Motivation

Machine learning workflows in protein science often suffer from **implicit assumptions**, **undocumented preprocessing**, **non-deterministic execution**, and **irreproducible data splits**.
While model architectures are extensively benchmarked, the *infrastructure that connects data, representations, training, and execution* is rarely externalized in a verifiable manner.

**MCS (Minimal Community Standard)** addresses this gap by providing a **minimal, tool-agnostic specification layer** that makes ML workflows:

* **auditable**
* **comparable**
* **reproducible**
* **portable across labs**

MCS is **not** a pipeline, framework, or benchmark.
It is a **standardized description layer** that externalizes the *decisions that matter*.

---

## Core idea

An ML run is fully determined by **five declarative specifications**:

| Spec            | Purpose                                               |
| --------------- | ----------------------------------------------------- |
| `DatasetSpec`   | What data is used, how it was curated, and from where |
| `SplitSpec`     | How data is partitioned and leakage is controlled     |
| `EmbeddingSpec` | How sequences are represented                         |
| `TrainSpec`     | What learning task is performed and how               |
| `ExecutionSpec` | Where and under which environment the run is executed |

From these specs, MCS generates:

* a **deterministic `run_id`**
* a **provenance record**
* a **local registry of specs and runs**

---

## Installation

```bash
git clone https://github.com/kren-ai-lab/MinimalCommunityStandar_v0.1.git
cd MinimalCommunityStandar_v0.1
pip install -e ".[dev]"
```

Requirements:

* Python ≥ 3.12
* `pydantic`, `pyyaml`
* `pytest` (for tests)

---

## Quickstart (Golden Path)

The **entire MCS workflow** can be executed in a few lines:

```python
import mcs
from mcs.provenance import ArtifactRef, MetricSummary

out = mcs.run_pack(
    "examples/",
    train_file="train_classification.yaml",
    execution_file="execution_gpu.yaml",
    strict=False,
    artifact_mode="reference",
    artifacts=[
        ArtifactRef(kind="metrics", path="artifacts/metrics.json"),
        ArtifactRef(kind="predictions", path="artifacts/preds.parquet"),
    ],
    metric_summary=MetricSummary(split="test", metrics={"roc_auc": 0.91}),
    notes="Baseline classification run",
)

print("Run ID:", out["record"].run_id)
print("Registry path:", out["registry_paths"]["run"])
```

This will:

1. Load the five YAML specs
2. Validate them (schema + semantic + cross-spec consistency)
3. Generate a deterministic `run_id`
4. Store specs and run metadata in a local registry

---

## Registry layout

After running MCS, a local registry is created:

```text
.mcs_registry/
├── specs/
│   ├── dataset/<fingerprint>.json
│   ├── split/<fingerprint>.json
│   ├── embedding/<fingerprint>.json
│   ├── train/<fingerprint>.json
│   └── execution/<fingerprint>.json
├── runs/
│   └── <run_id>.json
└── artifacts/
    └── <run_id>/...
```

This structure enables:

* run-level auditability
* comparison across experiments
* provenance-aware artifact tracking

---

## Validation philosophy

MCS validation operates at **three levels**:

1. **Schema validation**
   Enforced via `pydantic` (types, required fields)

2. **Semantic validation**
   Example:

   * supervised task requires labels
   * stratified split requires a valid field
   * CUDA execution without GPU is flagged

3. **Cross-spec consistency**
   Example:

   * embedding linked to a different dataset fingerprint
   * training split mismatch
   * missing leakage controls

Validation issues are classified as:

* `ERROR` → invalid configuration
* `WARNING` → potentially unsafe
* `INFO` → recommended improvements

---

## Provenance & lineage

Each run produces a **ProvenanceRecord** that links:

* spec fingerprints
* execution context
* produced artifacts
* summary metrics

Optionally, a **lineage graph** can be generated to visualize dependencies between specs, runs, and artifacts.

---

## Repository structure

```text
mcs/
├── api.py              # High-level facade (run_pack)
├── schemas/            # Dataset/Split/Embedding/Train/Execution specs
├── validation/         # Semantic and cross-spec validators
├── provenance/         # Run records and lineage
├── registry/           # Local registry backend
├── version.py
examples/
├── dataset.yaml
├── split.yaml
├── embedding.yaml
├── train_classification.yaml
├── train_regression.yaml
├── execution.yaml
└── execution_gpu.yaml
tests/
└── pytest-based test suite
```

---

## Tests

Run all tests with:

```bash
pytest -q
```

The test suite verifies:

* public API imports
* example YAML validity
* registry creation via `run_pack`

---

## What MCS is **not**

* Not a training framework
* Not a benchmark
* Not a model zoo
* Not opinionated about ML libraries

MCS is **pure infrastructure**.

---

## Citation

If you use MCS in academic work, please cite the associated manuscript:

> *Infrastructure, Not Just Models: Towards Verifiable and Scalable Machine Learning for Protein Engineering*
> *(under submission)*

---

## Contributing

Contributions are welcome, particularly:

* additional validators
* new split or leakage patterns
* registry backends
* interoperability tools

Please open an issue or pull request.

---

## Contact

**KrenAI Lab**
Universidad de Magallanes, Chile
[krenai@umag.cl](mailto:krenai@umag.cl)
