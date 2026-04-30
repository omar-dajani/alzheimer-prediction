#!/bin/bash
# aws_download_from_s3.sh — Download ADNI data from S3 to EC2 instance
# Run AFTER aws_setup.sh and AFTER S3 upload is complete.
#
# IMPORTANT: Replace YOUR-BUCKET-NAME with your actual S3 bucket name.

set -e

BUCKET="s3://adni-pipeline-2026"
BASE="/home/ec2-user/ADNI_Project"

echo "============================================================"
echo "  Downloading ADNI data from S3"
echo "============================================================"

# Download DICOMs (for Phase 2)
echo "[1/4] Downloading DICOMs from S3..."
aws s3 sync "${BUCKET}/ADNI_03.22.2026/" "${BASE}/data/raw/ADNI_03.22.2026/" --only-show-errors

# Download existing NIfTIs (for Phase 3.1 — the original 4,016 pre-processed files)
echo "[2/4] Downloading existing NIfTIs from S3..."
aws s3 sync "${BUCKET}/ADNI/" "${BASE}/data/raw/ADNI/" --only-show-errors

# Download metadata and support files
echo "[3/4] Downloading metadata files..."
aws s3 cp "${BUCKET}/adni_all_mri_metadata.csv" "${BASE}/data/raw/adni_all_mri_metadata.csv"
aws s3 cp "${BUCKET}/gold_standard_histogram.npy" "${BASE}/data/processed/gold_standard_histogram.npy"
aws s3 cp "${BUCKET}/ADNIMERGE.csv" "${BASE}/data/raw/ADNIMERGE.csv"
aws s3 cp "${BUCKET}/DXSUM_22Jan2026.csv" "${BASE}/data/raw/DXSUM_22Jan2026.csv"

# Download pipeline scripts
echo "[4/4] Copying pipeline scripts..."
# (These should already be on the instance if you SCP'd them, but just in case)

echo ""
echo "============================================================"
echo "  Download complete!"
echo "  DICOMs:   ${BASE}/data/raw/ADNI_03.22.2026/"
echo "  NIfTIs:   ${BASE}/data/raw/ADNI/"
echo "  Metadata: ${BASE}/data/raw/"
echo "============================================================"
