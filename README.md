# Minimal Community Standard (MCS)

*A working prototype for describing reproducible machine learning workflows in protein science*

> **Project status**
>
> MCS is an early-stage, research-oriented implementation of a lightweight workflow-record system. It is being developed to explore how architectural good practices for reproducible benchmarking can be operationalized in protein machine learning. At this stage, MCS should be understood as a draft coordination layer, not as a finalized community standard, mandatory framework, or replacement for existing tools.

---

## Motivation

Machine learning workflows in protein science often depend on **implicit assumptions**, **undocumented preprocessing**, **non-deterministic execution**, and **poorly documented data splits**. While model architectures are extensively benchmarked, the infrastructure that connects data, representations, training, execution, and evaluation is often not externalized in a verifiable manner.

MCS explores this gap by providing a **minimal, tool-agnostic specification layer** for recording the workflow decisions that shape machine learning results. The goal is to make protein ML workflows easier to:

* audit
* compare
* reproduce
* reuse across laboratories

MCS is **not** a pipeline, benchmark, model zoo, or training framework. It is a lightweight description layer for making workflow-defining assumptions explicit.

---

## Core idea

An ML run can be described through five declarative specifications:

| Spec            | Purpose                                               |
| --------------- | ----------------------------------------------------- |
| `DatasetSpec`   | What data is used, how it was curated, and from where |
| `SplitSpec`     | How data is partitioned and leakage is controlled     |
| `EmbeddingSpec` | How sequences are represented                         |
| `TrainSpec`     | What learning task is performed and how               |
| `ExecutionSpec` | Where and under which environment the run is executed |

From these specs, the current MCS prototype generates:

* a deterministic `run_id`
* a provenance record
* a local registry of specs and runs

These components are intended to support traceability across datasets, representations, splits, models, execution environments, and evaluation outputs.

---

## Installation

```bash
git clone https://github.com/kren-ai-lab/MinimalCommunityStandar_v0.1.git
cd MinimalCommunityStandar_v0.1
pip install -e ".[dev]"
````

Requirements:

* Python ≥ 3.12
* `pydantic`, `pyyaml`
* `pytest` for tests

---

## Quickstart

A basic MCS workflow can be executed with:

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

1. Load the five YAML specifications
2. Validate them using schema, semantic, and cross-spec checks
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

This structure supports:

* run-level auditability
* comparison across experiments
* provenance-aware artefact tracking

---

## Validation philosophy

MCS validation currently operates at three levels.

### 1. Schema validation

Basic structural validation is enforced through `pydantic`, including types and required fields.

### 2. Semantic validation

Examples include:

* supervised tasks require labels
* stratified splits require a valid stratification field
* CUDA execution without GPU availability is flagged

### 3. Cross-spec consistency

Examples include:

* embedding records linked to a different dataset fingerprint
* training split mismatch
* missing leakage-control metadata

Validation issues are classified as:

* `ERROR` for invalid configurations
* `WARNING` for potentially unsafe configurations
* `INFO` for recommended improvements

---

## Provenance and lineage

Each run produces a `ProvenanceRecord` linking:

* spec fingerprints
* execution context
* produced artefacts
* summary metrics

Optionally, a lineage graph can be generated to visualize dependencies between specs, runs, and artefacts.

---

## Repository structure

```text
mcs/
├── api.py              # High-level facade: run_pack
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
* registry creation through `run_pack`

---

## What MCS is not

At this stage, MCS is not:

* a finalized community standard
* a training framework
* a benchmark
* a model zoo
* an execution engine
* a replacement for workflow systems such as Nextflow, Snakemake, or CWL
* opinionated about machine learning libraries

MCS is currently a lightweight research prototype for recording workflow assumptions and supporting reproducible benchmarking experiments.

---

## Citation

If you use this prototype in academic work, please cite the associated manuscript when available:

> *Architectural Good Practices for Reproducible Benchmarking in Protein Machine Learning*
> under submission

---

## Contributing

Contributions are welcome, especially those related to:

* additional validators
* split and leakage-control patterns
* registry backends
* interoperability with workflow engines
* provenance visualization
* documentation and examples

Please open an issue or pull request.

---

## Contact

**KrenAI Lab**
Universidad de Magallanes, Chile
[krenai@umag.cl](mailto:krenai@umag.cl)

