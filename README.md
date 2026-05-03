# N.E.U.R.O.N.
### Neuroimaging and Event-based Unified Risk Outcomes Network
#### Rice University Capstone — Spring 2026

A survival analysis pipeline for predicting Alzheimer's disease progression using longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The pipeline predicts two clinical transitions: **MCI → Alzheimer's Dementia** and **CN → MCI or AD**, using four model families with full hyperparameter optimization, bootstrap confidence intervals, and cross-cohort evaluation.

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
├── Modeling on the Tabular dataset/
│   ├── notebook509862e120__5_.ipynb   # Main pipeline notebook
│   ├── modeling.py                    # Model training (Cox PH, GBSA, AFT, DeepSurv)
│   ├── preprocessing.py               # Harmonization, imputation, feature engineering
│   ├── postprocessing.py              # KM curves, survival curve plots
│   ├── concordance.py                 # IPCW time-dependent C-td implementation
│   ├── config.py                      # Shared path and constant configuration
│   ├── requirements.txt               # Python dependencies
│   ├── checkpoints/                   # Saved model checkpoints (.pkl) — not committed
│   ├── figures/                       # Generated plots saved during notebook execution
│   └── outputs/                       # model_comparison.csv and other result tables
├── EDA/                               # Exploratory data analysis
├── Transformer/                       # Transformer-based model experiments
└── README.md
```

---

## Data

> **⚠️ ADNI data is not included in this repository.** Access requires an approved application under the ADNI Data Use Agreement. Do not commit data files to this repo.

### Applying for access

Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu). Approval typically takes 1–2 weeks.

### What data this pipeline uses

The pipeline uses a single merged CSV built from the ADNIMERGE R package, which aggregates data from all ADNI study phases (ADNI1, ADNI GO, ADNI2, ADNI3, ADNI4). The following tables were exported from R and merged on `RID` (participant ID) and `VISCODE` (standardized visit code):

| Table | Contents |
|-------|----------|
| `adrs.csv` | ADAS-Cog cognitive assessment scores |
| `biomarkers.csv` | CSF (Amyloid-β, Tau, Phospho-Tau) and PET (FDG, AV45) measurements |
| `subjects.csv` | Demographics, diagnosis labels, APOE genotype |
| `UCSFFSX7.csv` | Structural MRI volumes from FreeSurfer segmentation |

The dataset used in this pipeline covers **2,430 baseline subjects** across CN, MCI, and AD diagnoses, with longitudinal follow-up of up to 10+ years.

### Placing the data

Once you have built the merged CSV, place it anywhere on your machine and update `DATA_PATH` in the path configuration cell of the notebook (Section 1.2).

---

## Setup

### Prerequisites

- Python 3.9 or later
- pip or conda
- The merged ADNI CSV (see Data section above)
- A GPU is optional but speeds up DeepSurv training (~15–30 min on CPU vs ~5 min on GPU)

### Installation

**Option A — pip + virtualenv**

```bash
git clone https://github.com/omar-dajani/alzheimer-prediction.git
cd alzheimer-prediction/Modeling\ on\ the\ Tabular\ dataset
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda**

```bash
git clone https://github.com/omar-dajani/alzheimer-prediction.git
cd alzheimer-prediction/Modeling\ on\ the\ Tabular\ dataset
conda create -n neuron python=3.10
conda activate neuron
pip install -r requirements.txt
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `lifelines` | Cox PH, Weibull AFT, Kaplan-Meier |
| `scikit-survival` | Gradient boosting survival analysis |
| `pycox` + `torchtuples` | DeepSurv neural Cox model |
| `neuroCombat` | MRI scanner batch effect correction |
| `optuna` | Bayesian hyperparameter optimization |
| `shap` | SHAP feature attribution for AFT |
| `scikit-learn` | MICE imputation, stratified splits |
| `torch` | DeepSurv neural network backend |

> If `neuroCombat` fails to install, try `pip install neuroCombat-sklearn` as an alternative.

---

## Running the Pipeline

### 1. Configure paths

In the notebook, find the path configuration cell (Section 1.2) and set:

```python
REPO_DIR  = Path('/path/to/alzheimer-prediction')
DATA_PATH = Path('/path/to/your/merged_adni.csv')
```

These are the only two lines you need to change.

### 2. Set RETRAIN flag

Near the top of the imports cell (Section 1.3):

```python
RETRAIN = True   # Train all models from scratch (~1–2 hours on CPU)
RETRAIN = False  # Load from saved checkpoints (seconds)
```

### 3. Run the notebook

Open the notebook in Jupyter, VS Code, or any environment that supports `.ipynb` files and run cells sequentially. The notebook is self-contained — all imports, data loading, model training, and visualization happen in order.

```bash
jupyter notebook "notebook509862e120__5_.ipynb"
# or
jupyter lab "notebook509862e120__5_.ipynb"
```

### Outputs

After a full run, the following are saved automatically:

| Location | Contents |
|----------|----------|
| `figures/` | Feature importance charts, KM quartile plots, individual survival curves, SHAP plots, Optuna diagnostics |
| `outputs/model_comparison.csv` | Final ranked model comparison table with C-td and 95% CIs |
| `checkpoints/` | Serialized model objects for each trained model — load with `RETRAIN = False` |

---

## Pipeline Overview

### Cohorts

| Cohort | Transition | Subjects | Events | Event rate |
|--------|-----------|---------|--------|------------|
| MCI | MCI → Alzheimer's Dementia | 958 | 385 | 40.2% |
| CN | CN → MCI or AD | 824 | 146 | 17.7% |

### Data Processing

1. **Diagnosis harmonization** — Remaps ADNI diagnosis variants (EMCI, LMCI, SMC, Dementia) to three canonical states (CN, MCI, AD)
2. **Reversion removal** — Excludes MCI subjects who reverted to CN per sponsor guidance, classified into trajectory groups (transient noise, sustained recovery, bouncers, progressors)
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

Missingness flags, protocol dummies, and composite features that are linear combinations of included raw features are explicitly excluded.

### Models

| Model | Type | HPO | Key strength |
|-------|------|-----|-------------|
| Cox PH | Semi-parametric linear | Optuna 30 trials, elastic-net regularization | Interpretable log hazard ratios, minimal overfitting |
| GBSA | Tree-based non-linear | Optuna 40 trials, 5-fold CV | Captures non-linear threshold effects |
| Weibull AFT | Parametric | 5-fold CV penalizer grid search | Absolute time predictions, best on CN cohort |
| DeepSurv | Neural Cox PH | Optuna 25 trials, early stopping | Detects APOE4 interaction effects |

All models are evaluated on a held-out 20% test set stratified by event status. Bootstrap 95% CIs use 500 resamples with no model refit.

---

## Reproducibility

- All random seeds set via `RANDOM_SEED = 42` and passed explicitly to all models, CV splitters, and imputers
- Train/test split is fixed before any model sees data and never touched during HPO
- Model checkpoints saved after each training run — set `RETRAIN = False` to reproduce results from saved checkpoints without retraining
- `checkpoints/` is excluded from version control via `.gitignore` — regenerate by running with `RETRAIN = True`

---

## GitHub Rendering

If the notebook fails to render on GitHub due to Optuna widget metadata, run this once before pushing:

```python
import json, pathlib
nb_path = "notebook509862e120__5_.ipynb"
nb = json.loads(pathlib.Path(nb_path).read_text())
nb["metadata"].pop("widgets", None)
pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
```
