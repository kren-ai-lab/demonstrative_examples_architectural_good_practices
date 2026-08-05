# Demonstrative Protein Machine-Learning Workflow

This repository implements a reproducible demonstrative workflow for antimicrobial peptide (AMP) classification. It normalizes peptide data, defines two binary AMP tasks, characterizes the resulting datasets, audits similarity-aware generalization regimes, creates repeated partitions, extracts frozen ESM-2 embeddings, trains conventional classifiers, and produces the data and figures used for methodological-effects analysis.

## Requirements

- A clean Ubuntu installation. Linux is required because the pinned MMseqs2 package is not available through Conda on Windows.
- Conda installed before starting this tutorial.
- An NVIDIA GPU compatible with the configured PyTorch/CUDA environment.
- Network access to download the ESM-2 checkpoint, or a populated local cache containing it.
- Sufficient time and storage for clustering, embedding extraction, repeated partition generation, and model training.

## 1. Clone the repository

Clone the project and enter its root directory:

```bash
git clone https://github.com/kren-ai-lab/demonstrative_examples_architectural_good_practices.git
cd demonstrative_examples_architectural_good_practices
```

Run all remaining commands from this repository root.

## 2. Create the environment

Create and activate the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate building_ml_models
```

## 3. Check the input

The workflow expects the following UTF-8 CSV file:

```text
data/raw/peptide_activity_pivot.csv
```

It must contain the columns `sequence`, `Antimicrobial`, and `Toxic`. Check the file non-destructively by reading only its first five rows:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/peptide_activity_pivot.csv', nrows=5); print(df.columns.tolist())"
```

## 4. Preprocess the dataset

Create a manifest containing the raw file's checksum and basic CSV metadata. The command writes `metadata/raw_file_manifest.json`:

```bash
python src/create_raw_manifest.py \
  --input data/raw/peptide_activity_pivot.csv \
  --output metadata/raw_file_manifest.json \
  --sequence-column sequence \
  --encoding utf-8 \
  --separator "," \
  --incorporation-date "2026-07-30"
```

Normalize and validate the sequences. The principal output is `data/processed/peptide_activity_normalized.csv`:

```bash
python src/normalize_sequences.py \
  --input data/raw/peptide_activity_pivot.csv \
  --output data/processed/peptide_activity_normalized.csv \
  --config configs/normalization.yml \
  --report metadata/normalization_report.json \
  --overwrite
```

Generate JSON and CSV quality-control summaries under `metadata/`:

```bash
python src/generate_quality_control_summary.py \
  --input data/processed/peptide_activity_normalized.csv \
  --config configs/normalization.yml \
  --json-output metadata/quality_control_summary.json \
  --csv-output metadata/quality_control_summary.csv \
  --overwrite
```

Construct the two configured AMP classification tasks. The principal output is `data/processed/amp_task.csv`:

```bash
python src/build_amp_tasks.py \
  --input data/processed/peptide_activity_normalized.csv \
  --config configs/amp_tasks.yml \
  --output data/processed/amp_task.csv \
  --report metadata/amp_task_report.json \
  --overwrite
```

Characterize the task datasets and write dataset summaries under `results/summaries/` with an execution record under `metadata/`:

```bash
python src/characterize_amp_tasks.py \
  --input data/processed/amp_task.csv \
  --config configs/dataset_characterization.yml \
  --overwrite
```

The main preprocessing outputs are `data/processed/peptide_activity_normalized.csv`, `data/processed/amp_task.csv`, preprocessing and quality-control records under `metadata/`, and dataset summaries under `results/summaries/`.

## 5. Generate and audit homology groups

> **Warning:** MMseqs2 clustering can take substantial time and storage.

Generate homology groups for the configured H90, H70, H50, and H30 regimes, corresponding to minimum sequence identities of 90%, 70%, 50%, and 30%:

```bash
python src/run_mmseqs_homology_clustering.py \
  --input data/processed/amp_task.csv \
  --config configs/homology_clustering.yml \
  --output-root data/homology \
  --overwrite
```

Audit the groups to check whether valid group-aware train, validation, and test partitions can be constructed:

```bash
python src/audit_homology_configurations.py \
  --config configs/homology_audit.yml \
  --overwrite
```

This stage writes homology data under `data/homology/`, candidate assignments under `results/homology_audit/`, summary tables under `results/summaries/`, and related records under `metadata/`.

## 6. Generate partitions

Generate repeated RANDOM, H90, H70, H50, and H30 train/validation/test assignments:

```bash
python src/build_partitions.py \
  --config configs/partitioning.yml \
  --overwrite
```

The assignments are written under `data/partitions/`; the partition index and validation summary are written under `results/summaries/`, with a generation record under `metadata/`.

## 7. Generate ESM-2 embeddings

> **Warning:** This stage requires the configured NVIDIA GPU, may download the ESM-2 checkpoint, may take substantial time, and writes a large Parquet representation file.

Extract the configured frozen ESM-2 embeddings with Sylphy:

```bash
python src/generate_sylphy_embeddings.py \
  --input data/processed/amp_task.csv \
  --config configs/representation.yml \
  --overwrite
```

The main output is `representations/amp_sylphy_embeddings.parquet`, with a representation record under `metadata/`.

## 8. Train and evaluate models

First validate one small grid configuration without fitting models:

```bash
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid \
  --tasks amp_vs_toxic_without_amp_evidence \
  --seeds 0 \
  --models logistic_regression linear_svm \
  --dry-run
```

Run the corresponding small model-training check:

```bash
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid \
  --tasks amp_vs_toxic_without_amp_evidence \
  --seeds 0 \
  --models logistic_regression linear_svm
```

> **Warning:** The complete grid and final evaluation run many repeated experiments and may take substantial time.

Run the complete model grid:

```bash
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid
```

Select the configured model variants:

```bash
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage select
```

Fit and evaluate the final models:

```bash
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage final
```

The main outputs are run data under `results/modeling/`, model-selection and performance tables under `results/summaries/`, and training records under `metadata/`.

## 9. Generate analysis tables and figures

Execute the notebooks in this order:

1. `notebooks/01_methodological_effects_analysis.ipynb`
2. `notebooks/02_figure2_methodological_effects_v8.ipynb`

Open them from the repository root or from `notebooks/` in a Jupyter-compatible editor such as VS Code, and select the `building_ml_models` environment as the kernel. The first notebook writes analysis tables under `results/final_analysis/`; the second writes final figures under `results/figures/`. Both write analysis or figure records under `metadata/`.

## 10. Run the complete workflow

After reviewing the individual stages, run all command-line stages with:

```bash
bash command_lines_executed.sh
```

This script includes the expensive clustering, embedding, and repeated training stages.

## Citation and license

Citation information is available in CITATION.cff. This project is licensed under the GNU General Public License v3.0; see LICENSE for details.
