#!/bin/bash
# run_pipeline_aws.sh — Run the full ADNI preprocessing pipeline on AWS EC2
# Runs Phase 2, 3.1, and 3.2 sequentially, then uploads results to S3.
#
# Usage: nohup bash run_pipeline_aws.sh > pipeline.log 2>&1 &
#        (use nohup so it survives SSH disconnects)
#
# IMPORTANT: Replace YOUR-BUCKET-NAME with your actual S3 bucket name.

set -e

BUCKET="s3://adni-pipeline-2026"
BASE="/home/ec2-user/ADNI_Project"
SCRIPTS="${BASE}/scripts"

# Activate conda environment
source activate adni || conda activate adni

# Restrict C++ threading libraries to 1 thread per Python worker.
# ANTsPy (ITK), deepbet (PyTorch), and scipy all spawn threads equal to
# the number of available cores by default. With N multiprocessing workers
# each spawning M threads, you get N*M threads fighting for M cores,
# causing severe context-switching overhead. Setting these to 1 lets
# Python's multiprocessing handle all parallelism cleanly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

echo "============================================================"
echo "  FULL PIPELINE: Phase 2 + 3.1 + 3.2 + 3.3 (AWS)"
echo "  Start: $(date)"
echo "============================================================"
echo ""

# Phase 2: DICOM-to-NIfTI Conversion
echo "[STEP 1/4] Phase 2: DICOM-to-NIfTI Conversion..."
python "${SCRIPTS}/phase2_dicom_to_nifti_aws.py"
echo ""

# Phase 3.1: NIfTI to 128^3 Tensor Conversion
echo "[STEP 2/4] Phase 3.1: NIfTI to 128^3 Tensor Conversion..."
python "${SCRIPTS}/01_mri_prep_improved_v2_aws.py"
echo ""

# Phase 3.2: Tabular Merge (Master CSV v2)
echo "[STEP 3/4] Phase 3.2: Tabular Merge (Master CSV v2)..."
python "${SCRIPTS}/02_tabular_prep_improved_v2_aws.py"
echo ""

# Phase 3.3: Longitudinal Deformation Fields (Flow Tensors)
echo "[STEP 4/4] Phase 3.3: Longitudinal Deformation Fields..."
python "${SCRIPTS}/phase3_3_generate_flows.py"
echo ""

echo "============================================================"
echo "  PIPELINE COMPLETE: $(date)"
echo "============================================================"
echo ""

# Upload results to S3 (only the final outputs you need)
echo "Uploading results to S3..."

echo "[1/3] Uploading MRI tensors..."
aws s3 sync "${BASE}/data/processed/mri_tensors_improved_v2/" \
    "${BUCKET}/results/mri_tensors_improved_v2/" --only-show-errors

echo "[2/3] Uploading flow tensors..."
aws s3 sync "${BASE}/data/processed/flow_tensors_v2/" \
    "${BUCKET}/results/flow_tensors_v2/" --only-show-errors

echo "[3/3] Uploading master CSVs and logs..."
aws s3 cp "${BASE}/data/processed/master_data_improved_04052026_v2.csv" \
    "${BUCKET}/results/master_data_improved_04052026_v2.csv"
aws s3 cp "${BASE}/data/processed/master_data_improved_04052026_v3.csv" \
    "${BUCKET}/results/master_data_improved_04052026_v3.csv"
aws s3 cp "${BASE}/data/processed/phase2_conversion_log.json" \
    "${BUCKET}/results/phase2_conversion_log.json" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  ALL DONE! Results uploaded to: ${BUCKET}/results/"
echo "  You can now download from S3 to your local PC."
echo "  Finish: $(date)"
echo "============================================================"
