# preprocessing data 

python src/create_raw_manifest.py \
  --input data/raw/peptide_activity_pivot.csv \
  --output metadata/raw_file_manifest.json \
  --sequence-column Sequence \
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

# task preparation
python src/build_amp_tasks.py \
  --input data/processed/peptide_activity_normalized.csv \
  --config configs/amp_tasks.yml \
  --output data/processed/amp_task.csv \
  --report metadata/amp_task_report.json \
  --overwrite

# data characterization
python src/characterize_amp_tasks.py \
  --input data/processed/amp_task.csv \
  --config configs/dataset_characterization.yml \
  --overwrite

# homology process 
python src/run_mmseqs_homology_clustering.py \
  --input data/processed/amp_task.csv \
  --config configs/homology_clustering.yml \
  --output-root data/homology
  --overwrite

# homology audit
python src/audit_homology_configurations.py \
  --config configs/homology_audit.yml \
  --overwrite

# preparing partitions
python src/build_partitions.py \
  --config configs/partitioning.yml \
  --overwrite