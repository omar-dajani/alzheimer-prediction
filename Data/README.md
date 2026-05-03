# Data Directory

This directory contains scripts and pipelines used to **download, preprocess, and construct** the datasets used in this project based on the Alzheimer's Disease Neuroimaging Initiative (ADNI).

---

## Data Access Notice

This repository uses ADNI data, which is subject to a **Data Use Agreement (DUA)**.

* Raw and processed data are **not stored directly in this repository**
* Access is restricted to **authorized users only**
* All users must obtain access via:

https://adni.loni.usc.edu/

---

## Directory Structure

```
Data/
├── Download_Data/
│   ├── download_tabular_dataset.py
│   ├── download_imaging_dataset.py
│   └── download_entire_master_dataset.py
│
└── Data Preprocessing Pipeline/
    ├── 01_mri_prep_improved_v2_aws.py
    ├── 02_tabular_prep_improved_v2_aws.py
    ├── phase2_dicom_to_nifti_aws.py
    ├── phase3_3_generate_flows.py
    ├── aws_download_from_s3.sh
    ├── aws_setup.sh
    └── run_pipeline_aws.sh
```

---

## Download_Data (Quick Access)

This folder provides scripts for downloading **preprocessed datasets** used for rapid experimentation and reproducibility.

### Scripts

* `download_tabular_dataset.py`
  Downloads the processed master tabular dataset (clinical + biomarkers + MRI references)

* `download_imaging_dataset.py`
  Downloads MRI tensors and longitudinal flow tensors

* `download_entire_master_dataset.py`
  Downloads all datasets (tabular + imaging)

---

### Usage

```bash
cd Data/Download_Data
python download_entire_master_dataset.py
```

---

### Outputs

| Dataset        | Description                                                           |
| -------------- | --------------------------------------------------------------------- |
| Master Dataset | Merged ADNI dataset with clinical, biomarker, and MRI-linked features |
| MRI Tensors    | Preprocessed 3D brain volumes                                         |
| Flow Tensors   | Longitudinal deformation fields capturing disease progression         |

---

## Data Preprocessing Pipeline (Full Reproducibility)

This pipeline enables full reconstruction of the dataset from raw ADNI data.

---

### Pipeline Overview

```
DICOM → NIfTI → MRI Tensors → Master Dataset → Flow Tensors
```

---

### Pipeline Steps

#### 1. DICOM → NIfTI Conversion

**Script:** `phase2_dicom_to_nifti_aws.py`

* Converts raw MRI DICOM scans to standardized NIfTI format
* Applies:

  * Registration
  * Bias correction
  * Intensity normalization

---

#### 2. MRI Tensor Generation

**Script:** `01_mri_prep_improved_v2_aws.py`

* Converts NIfTI volumes into 128×128×128 tensors
* Includes:

  * Skull stripping
  * Cropping and padding
  * Intensity normalization

---

#### 3. Tabular Dataset Construction

**Script:** `02_tabular_prep_improved_v2_aws.py`

* Merges:

  * ADNIMERGE
  * Diagnosis tables (DXSUM)
  * MRI metadata
* Outputs a master dataset linking:

  * patient IDs
  * diagnosis
  * imaging data

---

#### 4. Longitudinal Flow Generation

**Script:** `phase3_3_generate_flows.py`

* Computes deformation fields between visits
* Captures **disease progression over time**
* Outputs 3D flow tensors

---

### Running the Pipeline

#### 1. Setup environment

```bash
bash aws_setup.sh
```

#### 2. Download raw data

```bash
bash aws_download_from_s3.sh
```

#### 3. Run full pipeline

```bash
bash run_pipeline_aws.sh
```

---

### Final Outputs

| Output         | Description                              |
| -------------- | ---------------------------------------- |
| MRI Tensors    | 3D brain volumes for modeling            |
| Flow Tensors   | Longitudinal progression representations |
| Master Dataset | Fully integrated multimodal dataset      |

---

## Notes

* The `Download_Data` scripts are intended for **authorized users only**
* The preprocessing pipeline is the **source of truth** for dataset construction
* Outputs are generated locally and are not committed to the repository

---

## Public Release Note (Future)

When this repository is made public:

* The `Download_Data` directory will be removed
* Only pipeline code and reproduction instructions will remain
* Users will be required to obtain ADNI data independently

---

## Summary

This directory provides:

* Quick dataset access for reproducibility (private use)
* A full end-to-end pipeline for dataset construction
* Support for multimodal and longitudinal modeling

All data usage complies with ADNI access restrictions.
