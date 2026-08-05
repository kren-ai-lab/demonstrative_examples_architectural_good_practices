# Preprocess data

python src/create_raw_manifest.py \
  --input data/raw/peptide_activity_pivot.csv \
  --output metadata/raw_file_manifest.json \
  --sequence-column sequence \
  --encoding utf-8 \
  --separator "," \
  --incorporation-date "2026-07-30"

python src/normalize_sequences.py \
  --input data/raw/peptide_activity_pivot.csv \
  --output data/processed/peptide_activity_normalized.csv \
  --config configs/normalization.yml \
  --report metadata/normalization_report.json \
  --overwrite

python src/generate_quality_control_summary.py \
  --input data/processed/peptide_activity_normalized.csv \
  --config configs/normalization.yml \
  --json-output metadata/quality_control_summary.json \
  --csv-output metadata/quality_control_summary.csv \
  --overwrite

# Prepare classification tasks
python src/build_amp_tasks.py \
  --input data/processed/peptide_activity_normalized.csv \
  --config configs/amp_tasks.yml \
  --output data/processed/amp_task.csv \
  --report metadata/amp_task_report.json \
  --overwrite

# Characterize datasets
python src/characterize_amp_tasks.py \
  --input data/processed/amp_task.csv \
  --config configs/dataset_characterization.yml \
  --overwrite

# Cluster homologous sequences
python src/run_mmseqs_homology_clustering.py \
  --input data/processed/amp_task.csv \
  --config configs/homology_clustering.yml \
  --output-root data/homology \
  --overwrite

# Audit homology configurations
python src/audit_homology_configurations.py \
  --config configs/homology_audit.yml \
  --overwrite

# Build partitions
python src/build_partitions.py \
  --config configs/partitioning.yml \
  --overwrite

# Extract embeddings (requires the configured GPU/CUDA environment)
python src/generate_sylphy_embeddings.py \
  --input data/processed/amp_task.csv \
  --config configs/representation.yml \
  --overwrite

# Validate a small model-training configuration without training
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid \
  --tasks amp_vs_toxic_without_amp_evidence \
  --seeds 0 \
  --models logistic_regression linear_svm \
  --dry-run

# Run a small model-training check
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid \
  --tasks amp_vs_toxic_without_amp_evidence \
  --seeds 0 \
  --models logistic_regression linear_svm

# Run the full model grid
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage grid

# Select model variants
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage select

# Fit and evaluate final models
python src/train_amp_models.py \
  --config configs/modelling.yml \
  --stage final
