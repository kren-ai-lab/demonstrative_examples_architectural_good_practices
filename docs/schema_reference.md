## **Minimal Community Standard (MCS) – Schema Reference**

This document describes the **five core specification schemas** of the Minimal Community Standard (MCS).
Each schema is expressed as a declarative YAML file and validated using a combination of **structural, semantic, and cross-spec consistency rules**.

The goal of these schemas is **not to enforce a specific toolchain**, but to externalize the *minimal assumptions required to reproduce and audit an ML experiment*.

---

## Dataset Specification (`DatasetSpec`)

### Purpose

Declare **what data is used**, **where it comes from**, and **how it was curated**.

### Required fields

| Field         | Type   | Description                      |
| ------------- | ------ | -------------------------------- |
| `mcs_version` | string | MCS schema version               |
| `name`        | string | Dataset identifier               |
| `sources`     | list   | Provenance sources (≥1 required) |

### Optional but recommended

| Field                      | Description                        |
| -------------------------- | ---------------------------------- |
| `description`              | Human-readable dataset description |
| `domain`                   | e.g. `protein`, `peptide`          |
| `created_by`, `created_at` | Dataset authorship                 |
| `filters`                  | Declarative filtering rules        |
| `redundancy_control`       | Deduplication policy               |
| `labels`                   | Supervised targets                 |
| `metadata`                 | Auxiliary fields                   |
| `output`                   | Materialized dataset reference     |

### Example

```yaml
name: "PETase_classification_v1"
domain: "protein"

sources:
  - type: database
    name: UniProt
    version: "2024_01"
    uri: https://www.uniprot.org

filters:
  - field: sequence_length
    op: between
    value: [100, 600]

redundancy_control:
  method: sequence_identity
  threshold: 0.9
  tool: mmseqs2

labels:
  - name: activity
    dtype: bool
```

---

## Split Specification (`SplitSpec`)

### Purpose

Define **how data is partitioned** and **how leakage is controlled**.

### Core fields

| Field       | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `protocol`  | `random`, `stratified`, `group`, `temporal`, `cluster_aware` |
| `fractions` | Train/val/test proportions                                   |

### Protocol-specific fields

| Protocol        | Required fields  |
| --------------- | ---------------- |
| `stratified`    | `stratify_field` |
| `group`         | `group_field`    |
| `temporal`      | `time_field`     |
| `cluster_aware` | `identity`       |

### Leakage checks

Leakage checks are **declarative assertions** evaluated post-split.

```yaml
leakage_checks:
  - type: group_overlap
    field: protein_family
  - type: label_distribution
    tolerance: 0.05
```

### Example

```yaml
protocol: stratified
stratify_field: activity

fractions:
  train: 0.7
  val: 0.15
  test: 0.15
```

---

## Embedding Specification (`EmbeddingSpec`)

### Purpose

Declare **how sequences are represented numerically**.

### Core fields

| Field     | Description                      |
| --------- | -------------------------------- |
| `method`  | `plm`, `onehot`, `custom`        |
| `model`   | Model identifier                 |
| `pooling` | `mean`, `cls`, `attention`, etc. |
| `layer`   | Embedding layer                  |

### Optional

| Field           | Description                 |
| --------------- | --------------------------- |
| `normalization` | e.g. `zscore`, `l2`         |
| `cache`         | Cache strategy              |
| `parameters`    | Free-form embedding options |

### Example

```yaml
method: plm
model: facebook/esm2_t33_650M
layer: -1
pooling: mean
normalization: l2
```

---

## Training Specification (`TrainSpec`)

### Purpose

Declare **what learning task is solved** and **how models are trained**.

MCS supports **task-specific schemas**, but they share a common core.

### Core fields

| Field       | Description                      |
| ----------- | -------------------------------- |
| `task`      | `classification` or `regression` |
| `algorithm` | Model family                     |
| `features`  | Input feature source             |
| `target`    | Label field                      |

### Classification example

```yaml
task: classification
algorithm: random_forest
target: activity

hyperparameters:
  n_estimators: 300
  max_depth: 12

metrics:
  - roc_auc
  - f1
```

### Regression example

```yaml
task: regression
algorithm: xgboost
target: melting_temperature

metrics:
  - rmse
  - r2
```

---

## Execution Specification (`ExecutionSpec`)

### Purpose

Externalize **the execution environment** in which results were produced.

### Core fields

| Field            | Description                 |
| ---------------- | --------------------------- |
| `platform`       | OS / architecture           |
| `python_version` | Python runtime              |
| `dependencies`   | Dependency capture strategy |
| `hardware`       | CPU/GPU declaration         |

### Determinism controls

```yaml
determinism_flags:
  python_hash_seed: 0
  numpy_seed: 42
  torch_seed: 42
  torch_deterministic: true
```

### GPU example

```yaml
hardware:
  cpu: Intel Xeon
  gpu: NVIDIA A100
  accelerator: cuda
  ram_gb: 256
```

---

## Cross-spec consistency

MCS validates **interactions between specs**, including:

* dataset ↔ split field coherence
* redundancy vs identity-aware splits
* embedding ↔ dataset fingerprint matching
* training ↔ task/label compatibility
* execution ↔ determinism expectations

These checks ensure that *valid YAML files still represent a scientifically coherent experiment*.

---

## Minimal compliance checklist

To be **MCS-compliant**, a run must provide:

* one `DatasetSpec` with ≥1 provenance source
* one `SplitSpec`
* one `EmbeddingSpec`
* one `TrainSpec`
* one `ExecutionSpec`

Everything else is optional.

---

## Design philosophy

* Declarative over procedural
* Minimal over exhaustive
* Verifiable over convenient
* Tool-agnostic by design

---