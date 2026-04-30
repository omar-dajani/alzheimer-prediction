#!/bin/bash
# aws_setup.sh — One-time environment setup on EC2 instance
# Run this ONCE after SSH-ing into the instance.

set -e

echo "============================================================"
echo "  AWS EC2 Setup for ADNI Pipeline"
echo "============================================================"

# 1. Install system dependencies
echo "[1/5] Installing system dependencies..."
sudo yum install -y dcm2niix || {
    echo "dcm2niix not in yum, installing from conda..."
    INSTALL_DCM2NIIX_VIA_CONDA=true
}

# 2. Create project directory structure
echo "[2/5] Creating directory structure..."
mkdir -p /home/ec2-user/ADNI_Project/data/raw
mkdir -p /home/ec2-user/ADNI_Project/data/processed/mri_tensors_improved_v2
mkdir -p /home/ec2-user/ADNI_Project/data/processed/flow_tensors_v2
mkdir -p /home/ec2-user/ADNI_Project/scripts

# 3. Create conda environment
echo "[3/5] Creating conda environment..."
conda create -n adni python=3.10 -y
source activate adni || conda activate adni

# 4. Install Python dependencies
echo "[4/5] Installing Python packages..."
pip install antspyx nibabel scipy pandas numpy scikit-image deepbet tqdm

# Install dcm2niix via conda if yum failed
if [ "${INSTALL_DCM2NIIX_VIA_CONDA}" = "true" ]; then
    conda install -c conda-forge dcm2niix -y
fi

# 5. Verify installation
echo "[5/5] Verifying installation..."
python -c "import ants; print('ANTsPy OK')"
python -c "import nibabel; print('nibabel OK')"
python -c "import deepbet; print('deepbet OK')"
python -c "from scipy.ndimage import zoom; print('scipy OK')"
python -c "import pandas; print('pandas OK')"
dcm2niix --version || echo "WARNING: dcm2niix not found in PATH"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "    1. Download data from S3:  bash aws_download_from_s3.sh"
echo "    2. Run pipeline:           bash run_pipeline_aws.sh"
echo "============================================================"
