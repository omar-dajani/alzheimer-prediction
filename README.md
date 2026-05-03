# N.E.U.R.O.N.
### Neuroimaging and Event-based Unified Risk Outcomes Network
#### Rice University D2K Capstone — Spring 2026

A survival analysis pipeline for predicting Alzheimer's disease progression using longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The pipeline predicts two clinical transitions — **MCI → Alzheimer's Dementia** and **CN → MCI or AD** — using four model families with full hyperparameter optimization, bootstrap confidence intervals, and cross-cohort evaluation.

**Team:** Nathon Chavez, Omar Dajani, Eliza Iqbal, Savannah Nix, Fabrizio Pacheco, Evie Roth, Shichen Tang  
**Sponsor:** Cindy Zhang
**Instructor:** Xinjie Lan
**Mentor:** Antonio Mendoza Gonzales

---

## Results

All metrics are IPCW Antolini time-dependent C-td on a held-out 20% test set with 500-resample bootstrap 95% CIs. A C-td above 0.75 is considered strong for Alzheimer's progression prediction at ADNI's censoring level.

| Model | MCI → AD C-td | 95% CI | CN → MCI/AD C-td | 95% CI |
|-------|--------------|--------|-----------------|--------|
| Cox PH | **0.8422** | [0.806, 0.884] | 0.7633 | [0.682, 0.849] |
| AFT (Weibull) | 0.8418 | [0.806, 0.884] | **0.7719** | [0.692, 0.849] |
| GBSA | 0.8286 | [0.777, 0.869] | 0.7287 | [0.643, 0.830] |
| DeepSurv | 0.8231 | [0.775, 0.863] | 0.7491 | [0.659, 0.842] |

**Recommended models:** Cox PH for MCI → AD (highest C-td, fully interpretable coefficients, near-zero overfitting gap). AFT for CN → MCI/AD (best CN performance, provides absolute time-to-event predictions).

---

## Repository Structure

```
alzheimer-prediction/
│
├── Data/
│   ├── Download_Data/                 # Download preprocessed datasets (authorized users)
│   │   ├── download_tabular_dataset.py        # Download merged tabular data
│   │   ├── download_imaging_dataset.py        # Download MRI + flow tensors
│   │   └── download_entire_master_dataset.py  # Download all datasets
│   │
│   └── Data Preprocessing Pipeline/   # Raw ADNI → modeling-ready pipeline
│       ├── 01_mri_prep_improved_v2_aws.py     # NIfTI → MRI tensors
│       ├── 02_tabular_prep_improved_v2_aws.py # Merge + build master table
│       ├── phase2_dicom_to_nifti_aws.py       # DICOM → NIfTI conversion
│       ├── phase3_3_generate_flows.py         # Generate longitudinal flows
│       ├── aws_download_from_s3.sh            # Download raw data from S3
│       ├── aws_setup.sh                      # Setup AWS environment
│       └── run_pipeline_aws.sh               # Run full pipeline
│
├── EDA/
│   ├── Multimodal vs Single Modality/        # Integrated multimodal dataset + coverage analysis
│   ├── Feature Deep Dive/                   # Feature-level distributions, correlations, and longitudinal behavior
│   ├── MRI_Longitudinal_Dynamics/           # MRI scan availability, retention, and dropout patterns
│   ├── Statistical and Patient Trajectories/ # ANOVA, feature significance, and patient-level trajectories
│   └── README.md                            # Overview of all EDA analyses
│
├── Modeling on the Imaging Dataset/   # Transformer-based survival model on MRI
│   ├── Config/        # Centralized model hyperparameters
│   ├── Datasets/          # Dataset + normalization logic
│   ├── Losses/        # Survival loss functions (IPCW)
│   ├── Metrics/       # Evaluation metrics (C_td, Uno C, etc.)
│   ├── Models/        # Core model architecture components
│   ├── Training/      # Training loop + optimization logic
│   ├── Utils/         # Supporting utilities (encoding, interpolation)
│   ├── WORKFLOW.md    # End-to-end pipeline explanation
│   ├── run_all_pipeline.ipynb  # Notebook runner for full pipeline
│   └── train.py       # CLI entry point for training
│
├── Modeling on the Tabular dataset/   # Main tabular survival pipeline (this README)
│   ├── Tabular_Survival_Analysis_Pipeline.ipynb   # Main notebook
│   ├── modeling.py                    # Cox PH, GBSA, AFT, DeepSurv training
│   ├── preprocessing.py               # Harmonization, imputation, feature engineering
│   ├── postprocessing.py              # KM curves, survival curve plots
│   ├── concordance.py                 # IPCW time-dependent C-td implementation
│   ├── config.py                      # Shared path and constant configuration
│   ├── checkpoints/                   # Saved model checkpoints (.pkl) — not committed
│   ├── figures/                       # Generated plots saved during notebook execution
│   ├── outputs/                       # model_comparison.csv and result tables
│   └── tables/                        # Data files — not committed (see Data section)
│
├── .gitignore
├── README.md
├── environment.yml
└── requirements.txt
```

---

## Data

> **This repository is currently private and contains ADNI-derived data access utilities.**  
Access is restricted to authorized users (team members and instructor) under the ADNI Data Use Agreement.

> **ADNI data is not included in this repository.**  
Access requires an approved application under the ADNI Data Use Agreement. Do not commit data files to this repo — they are excluded via `.gitignore`.

---

## Data Access Options

This project supports two workflows depending on your use case.

---

### Option 1 — Quick Start (Preprocessed Data)

The `Data/Download_Data/` directory provides scripts to download preprocessed datasets for rapid experimentation and reproducibility.

#### Scripts

- `download_tabular_dataset.py`  
  Downloads the merged ADNI tabular dataset (clinical + biomarkers + MRI references)

- `download_imaging_dataset.py`  
  Downloads MRI tensors and longitudinal flow tensors

- `download_entire_master_dataset.py`  
  Downloads all datasets

#### Usage

```bash
cd Data/Download_Data
python download_entire_master_dataset.py

---

## Setup

### Prerequisites

- Python 3.9 or later
- pip or conda
- The merged ADNI CSV (see Data section)
- A GPU is optional but speeds up DeepSurv training (~15–30 min on CPU vs ~5 min on GPU)

### Installation

**Option A — pip + virtualenv**

```bash
git clone https://github.com/omar-dajani/alzheimer-prediction.git
cd "alzheimer-prediction/Modeling on the Tabular dataset"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda**

```bash
git clone https://github.com/omar-dajani/alzheimer-prediction.git
cd "alzheimer-prediction/Modeling on the Tabular dataset"
conda env create -f environment.yml
conda activate neuron
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `lifelines` | Cox PH, Weibull AFT, Kaplan-Meier |
| `scikit-survival` | Gradient boosting survival analysis (GBSA) |
| `pycox` + `torchtuples` | DeepSurv neural Cox model |
| `neuroCombat` | MRI scanner batch effect correction |
| `optuna` | Bayesian hyperparameter optimization |
| `shap` | Feature attribution for AFT model |
| `scikit-learn` | MICE imputation, stratified splits |
| `torch` | DeepSurv neural network backend |

> If `neuroCombat` fails to install, try `pip install neuroCombat-sklearn` instead.

---

## Exploratory Data Analysis (EDA)

The `EDA/` directory contains a set of complementary analyses designed to build a complete understanding of the ADNI dataset before modeling.

Each EDA focuses on a different dimension of the data:

### Multimodal vs Single Modality
- Integrates genetics, plasma biomarkers, cognition, and MRI-derived features
- Analyzes overlap between modalities and cohort construction
- Highlights limitations of multimodal modeling due to sparse MRI coverage

### Feature Deep Dive
- Examines distributions and correlations of key clinical features
- Compares CN, MCI, and AD populations
- Validates biological signals (e.g., APOE, amyloid, tau)

### MRI Longitudinal Dynamics
- Studies MRI scan frequency, follow-up duration, and dropout behavior
- Identifies retention bias (progressing patients are tracked longer)
- Quantifies imaging vs tabular data availability gaps

### Statistical and Patient Trajectories
- Performs ANOVA-based feature significance testing
- Visualizes feature distributions across diagnosis groups
- Tracks individual patient trajectories over time

### Why this matters

These analyses collectively:

- Define the **true usable cohort**
- Identify **high-signal features**
- Reveal **longitudinal and modality biases**
- Inform **model design and feature engineering decisions**

The EDA work directly supports the survival modeling pipeline by ensuring that assumptions about the data are validated before training.

For full details, see:

```
EDA/README.md
```

---

## Running the Tabular Pipeline

### 1. Configure paths

In the Tabular_Survival_Analysis_Pipeline notebook, find the path configuration cell (Section 1.2) and set:

```python
REPO_DIR  = Path('/path/to/alzheimer-prediction')
DATA_PATH = Path('/path/to/your/merged_adni.csv')
```

These are the only two lines you need to change.

### 2. Set the RETRAIN flag

Near the top of the imports cell (Section 1.3):

```python
RETRAIN = True   # Train all models from scratch (~1–2 hours on CPU)
RETRAIN = False  # Load from saved checkpoints (seconds)
```

### 3. Run the notebook

Open `Tabular_Survival_Analysis_Pipeline.ipynb` in Jupyter, VS Code, or any environment that supports `.ipynb` files and run cells sequentially.

```bash
cd "alzheimer-prediction/Modeling on the Tabular dataset"
jupyter notebook Tabular_Survival_Analysis_Pipeline.ipynb
# or
jupyter lab Tabular_Survival_Analysis_Pipeline.ipynb
```

### Outputs

After a full run the following are saved automatically:

| Location | Contents |
|----------|----------|
| `figures/` | Feature importance charts, KM quartile plots, individual survival curves, SHAP plots, Optuna diagnostics |
| `outputs/model_comparison.csv` | Final ranked model comparison table with C-td and 95% CIs |
| `checkpoints/` | Serialized model objects — reload with `RETRAIN = False` to skip retraining |

---

## Pipeline Overview

### Cohorts

| Cohort | Transition | Subjects | Events | Event rate |
|--------|-----------|---------|--------|------------|
| MCI | MCI → Alzheimer's Dementia | 958 | 385 | 40.2% |
| CN | CN → MCI or AD | 824 | 146 | 17.7% |

### Data Processing

1. **Diagnosis harmonization** — Remaps ADNI diagnosis variants (EMCI, LMCI, SMC, Dementia) to three canonical states (CN, MCI, AD)
2. **Reversion removal** — Excludes MCI subjects who reverted to CN, classified into trajectory groups (transient noise, sustained recovery, bouncers, progressors)
3. **MRI batch effect correction** — ComBat harmonization removes 1.5T vs 3T scanner bias while preserving biological variance
4. **Tiered imputation** — Three-stage strategy: longitudinal nearest-neighbor fill → MICE → two-stage LightGBM CSF predictor for missing Amyloid-β
5. **Feature engineering** — ICV-normalized MRI volumes, APOE4 interaction terms, ratio features

### Features (33 per cohort)

| Category | Features |
|----------|---------|
| Demographics | Age, Sex, Education, APOE ε4 allele count |
| Cognitive tests | MMSE, CDR-SB, ADAS-Cog 11/13, RAVLT, FAQ, MoCA, ECog, mPACC |
| MRI volumes | Hippocampus, Entorhinal, Ventricles, Fusiform, MidTemporal, WholeBrain (all ICV-adjusted) |
| CSF / PET biomarkers | Amyloid-β, Total Tau, Phospho-Tau, FDG-PET, AV45-PET |
| APOE4 interactions | APOE4 × Amyloid Load, APOE4 × Tau Burden, APOE4 × Hippocampal Volume, APOE4 × Amyloid Positivity |

### Models

| Model | Type | HPO | Key strength |
|-------|------|-----|-------------|
| Cox PH | Semi-parametric linear | Optuna 30 trials, elastic-net regularization | Interpretable log hazard ratios, minimal overfitting |
| GBSA | Tree-based non-linear | Optuna 40 trials, 5-fold CV | Captures non-linear threshold effects |
| Weibull AFT | Parametric | 5-fold CV penalizer grid search | Absolute time-to-event predictions, best CN model |
| DeepSurv | Neural Cox PH | Optuna 25 trials, early stopping | Detects APOE4 × pathology interaction effects |

---

## Reproducibility

- All random seeds set via `RANDOM_SEED = 42` and passed explicitly to all models, CV splitters, and imputers
- Train/test split is fixed before any model sees data and never touched during HPO
- `checkpoints/` is excluded from version control — regenerate by running with `RETRAIN = True`
- ADNI data must not be committed per the ADNI Data Use Agreement — all CSV paths under `tables/` are in `.gitignore`

---

## GitHub Rendering

If the notebook fails to render on GitHub due to Optuna widget metadata, run this once before pushing:

```python
import json, pathlib
nb_path = "Tabular_Survival_Analysis_Pipeline.ipynb"
nb = json.loads(pathlib.Path(nb_path).read_text())
nb["metadata"].pop("widgets", None)
pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
```
