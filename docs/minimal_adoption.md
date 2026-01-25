# **Minimal Adoption Guide (MCS v0.1)**

*What is the minimum you need to be MCS-compliant?*

---

## Purpose

This guide defines the **minimal requirements** to adopt the Minimal Community Standard (MCS).

MCS is intentionally lightweight:
you do **not** need to change your training code, models, or infrastructure.

You only need to **declare what you already do**.

---

## The minimal contract

An experiment is **MCS-compliant** if it provides **five YAML specifications**:

| File             | Required | Purpose                             |
| ---------------- | -------- | ----------------------------------- |
| `dataset.yaml`   | ✔        | Data provenance and curation        |
| `split.yaml`     | ✔        | Data partitioning & leakage control |
| `embedding.yaml` | ✔        | Sequence representation             |
| `train.yaml`     | ✔        | Learning task & model               |
| `execution.yaml` | ✔        | Execution environment               |

Nothing else is mandatory.

---

## Minimal `dataset.yaml`

At minimum, a dataset must declare **where the data comes from**.

```yaml
name: "my_dataset"

sources:
  - type: database
    name: UniProt
```

This alone already:

* anchors provenance
* enables dataset fingerprinting
* prevents “floating datasets”

### Optional (recommended if applicable)

* filters
* redundancy control
* labels

---

## Minimal `split.yaml`

A valid split must specify a protocol and fractions.

```yaml
protocol: random

fractions:
  train: 0.8
  test: 0.2
```

That’s it.

> If your task is supervised, **stratified splits are recommended**, but not mandatory.

---

## Minimal `embedding.yaml`

At minimum, you must declare **how sequences are represented**.

```yaml
method: onehot
```

More expressive embeddings are optional:

```yaml
method: plm
model: facebook/esm2_t12_35M
pooling: mean
```

---

## Minimal `train.yaml`

Training must declare **what task is solved**.

### Classification

```yaml
task: classification
algorithm: random_forest
target: activity
```

### Regression

```yaml
task: regression
algorithm: linear_regression
target: stability
```

Hyperparameters, metrics, and optimization details are optional.

---

## Minimal `execution.yaml`

Execution must externalize **where the experiment ran**.

```yaml
platform: Linux
python_version: "3.12"
```

### Optional but recommended

* dependency snapshot
* hardware declaration
* determinism flags

---

## Validation behavior

MCS distinguishes between:

| Severity  | Meaning                               |
| --------- | ------------------------------------- |
| `ERROR`   | Invalid or inconsistent configuration |
| `WARNING` | Potentially unsafe or non-ideal       |
| `INFO`    | Recommendation or best practice       |

Only `ERROR` blocks execution.

---

## Example: minimal compliant experiment

```text
dataset.yaml
split.yaml
embedding.yaml
train.yaml
execution.yaml
```

With these files, you can already run:

```python
import mcs
mcs.run_pack("my_experiment/")
```

This will generate:

* a deterministic run identifier
* a provenance record
* a local registry entry

---

## What happens if you omit fields?

* Missing **required fields** → validation `ERROR`
* Missing **recommended fields** → `WARNING` or `INFO`
* Missing **optional fields** → silently accepted

MCS never forces completeness.

---

## Design rationale

The minimal adoption layer exists to:

* enable **incremental adoption**
* avoid disrupting existing workflows
* allow partial compliance
* support legacy experiments

MCS can be adopted **retroactively** to describe past experiments.
