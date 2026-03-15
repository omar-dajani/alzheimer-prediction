# Alzheimer's Disease Progression Prediction
### Rice University D2K Capstone — Spring 2026

A multistate survival analysis pipeline for predicting Alzheimer's disease progression using longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The pipeline covers data harmonization, tiered imputation, feature engineering, and baseline survival models (LightGBM and DeepSurv), with an ensemble layer combining model outputs.

---

## Team

**Students:** Nathon Chavez, Omar Dajani, Eliza Iqbal, Savannah Nix, Fabrizio Pacheco, Evie Roth, Schichen Tang

**Sponsor:** Antonio Mendoza Gonzales

---

## Repository Structure

```
alzheimer-prediction/
├── Baseline/
│   ├── ADNI_Survival_Pipeline.ipynb   # Main pipeline notebook
│   ├── modeling.py                    # Model training functions (LightGBM, DeepSurv, ensemble)
│   ├── preprocessing.py               # Data harmonization, imputation, feature engineering
│   ├── postprocessing.py              # Calibration plots, KM curves, subject-time matrix
│   ├── fix_notebook_error.py          # Strips Optuna widget metadata for GitHub rendering
│   ├── checkpoints/                   # Saved model checkpoints (.pkl)
│   ├── figures/                       # Generated plots and visualizations
│   ├── outputs/                       # Model output tables
│   └── tables/                        # Preprocessed data tables
├── EDA/                               # Exploratory data analysis notebooks and figures
├── Transformer/                       # Transformer-based model on MRI imaging
├── .gitignore
└── README.md
```

---

## Data Access

> **⚠️ The ADNI data files are not included in this repository due to the ADNI Data Use Agreement. You must apply for access independently.**

This pipeline uses a merged dataset combining ADNIMERGE and ADNI4 data.

**[PLACEHOLDER — to be completed by data pipeline lead]**

> Describe here: which ADNI tables were downloaded, how the merge was performed, what the final merged file is named, and where to place it in the directory before running the notebook.

To apply for ADNI data access, visit: [adni.loni.usc.edu](https://adni.loni.usc.edu)

Once access is granted and the data is prepared, place the merged CSV file at:
```
Baseline/tables/ADNIMERGE_28Feb2026.csv
```

---

## Installation

This pipeline runs in **Google Colab** with a GPU runtime. No local installation is required beyond cloning the repository.

### 1. Clone the repository

```bash
git clone https://github.com/omar-dajani/alzheimer-prediction.git
cd alzheimer-prediction/Baseline
```

### 2. Upload to Google Colab

Upload the `Baseline/` folder to your Google Drive, then open `ADNI_Survival_Pipeline.ipynb` in Colab.

### 3. Install dependencies

The first cell of the notebook installs all required packages. You can also install them manually:

```bash
pip install lifelines scikit-survival pycox lightgbm xgboost optuna shap \
            fancyimpute neuroCombat torchtuples torch numpy pandas \
            matplotlib seaborn scipy scikit-learn tqdm
```

### Key dependencies

| Package | Purpose |
|---|---|
| `pycox` + `torchtuples` | DeepSurv neural survival model |
| `lightgbm` | Gradient boosting survival model |
| `lifelines` | Cox PH, Kaplan-Meier, C-index |
| `neuroCombat` | MRI batch effect correction |
| `optuna` | Hyperparameter tuning |
| `shap` | Feature importance |
| `scikit-survival` | Survival analysis utilities |

> **GPU runtime required** for LightGBM (`device='gpu'`) and DeepSurv (PyTorch). In Colab: Runtime → Change runtime type → T4 GPU.

---

## Usage

### Running the full pipeline

Open `ADNI_Survival_Pipeline.ipynb` in Google Colab and run cells sequentially. The notebook is organized into clearly labeled sections:

| Cell | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Imports and global config |
| 3 | Data loading and DX harmonization |
| 4 | Reversion removal |
| 5 | Survival label construction |
| 6 | Batch effect / ComBat harmonization |
| 7 | Longitudinal slope computation |
| 8 | Tiered imputation (longitudinal fill → MICE → CSF) |
| 9 | Feature engineering |
| 10 | Cohort assembly and final imputation |
| 11 | Domain-separated feature sets |
| 12 | Evaluation framework setup |
| 13 | Two-stage CSF imputation |
| 16 | LightGBM survival model |
| 17 | DeepSurv neural Cox model |
| 20 | Ensemble strategies |
| 21–23 | Calibration, KM curves, model comparison table |
| 24 | Subject × time matrix for future transformer use |

### Retraining vs. loading from checkpoint

Set the `RETRAIN` flag at the top of the notebook:

```python
RETRAIN = True   # Train all models from scratch
RETRAIN = False  # Load from saved checkpoints in Baseline/checkpoints/
```

Checkpoints are saved automatically after each model trains. When `RETRAIN = False`, models are loaded from `.pkl` files in the `checkpoints/` directory.

---

## Notebook GitHub Rendering Fix

The notebook may fail to render on GitHub due to Optuna widget metadata left in the notebook file. To fix this before pushing, run `fix_notebook_error.py` from the `Baseline/` directory:

```python
# fix_notebook_error.py — run once before committing
import json, pathlib

nb_path = "ADNI_Survival_Pipeline.ipynb"
nb = json.loads(pathlib.Path(nb_path).read_text())
nb["metadata"].pop("widgets", None)
pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
```

This strips the widget metadata and rewrites the file cleanly. Then commit the cleaned notebook.

---

## Pipeline Overview

### Cohorts

| Cohort | Transition | N (approx.) | Event rate |
|---|---|---|---|
| MCI | MCI → Dementia | ~825 | ~50% |
| CN | CN → Decline | ~824 | ~18% |

### Imputation Strategy (3 tiers)

1. **Longitudinal nearest-neighbor fill** — uses visits within ±1 year per subject
2. **MICE** — multivariate imputation for remaining missingness
3. **Two-stage CSF prediction** — LightGBM model predicts missing ABETA from PET/MRI features

### Harmonization

ComBat batch effect correction is applied to MRI volumetric features (Hippocampus, Entorhinal, Ventricles, Fusiform, MidTemp, WholeBrain) to remove scanner field strength bias (1.5T vs. 3T) while preserving biological variance.

### Models

| Model | Cohort | Test C-index |
|---|---|---|
| LightGBM Survival | MCI → Dementia | [PLACEHOLDER] |
| LightGBM Survival | CN → Decline | [PLACEHOLDER] |
| DeepSurv | MCI → Dementia | [PLACEHOLDER] |
| DeepSurv | CN → Decline | [PLACEHOLDER] |
| Ensemble | MCI → Dementia | [PLACEHOLDER] |
| Ensemble | CN → Decline | [PLACEHOLDER] |

> Results will be populated upon final model runs.

---

## Module Reference

### `preprocessing.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `classify_reverters(df_all)` | Full longitudinal DataFrame | Dict of RID sets by group | Classifies MCI→CN reverters into trajectory groups |
| `build_survival_labels(df_all, df_baseline, from_dx, to_dx)` | Longitudinal + baseline DFs, DX strings | DataFrame with event/duration per RID | Constructs survival labels for a given cohort transition |
| `run_combat(df_baseline)` | Baseline DataFrame | Harmonized baseline DataFrame | Applies ComBat MRI batch correction |
| `compute_slopes_cutoff(df_all, surv_labels, features)` | Longitudinal DF, survival labels, feature list | DataFrame of per-subject OLS slopes | Computes leakage-free longitudinal slopes |
| `longitudinal_fill(df_all, features)` | Longitudinal DF, feature list | Filled DataFrame | Nearest-neighbor longitudinal imputation (Tier 1) |
| `mice_impute(X_df)` | Feature DataFrame | Imputed DataFrame | MICE imputation via IterativeImputer (Tier 2) |
| `assemble_cohort(df_baseline, surv_labels, slopes_df, ...)` | Multiple DFs | X, y_event, y_duration, RIDs | Merges baseline + slopes + labels into model-ready cohort |
| `get_domain_features(feature_names)` | List of feature names | Dict of domain → feature lists | Separates features into imaging, CSF/PET, and cognitive domains |

### `modeling.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `lgb_survival_cv(X_imp, y_event, y_duration, feature_names, label)` | Feature matrix, survival labels, feature names | C-index, importance series, fitted model | Optuna-tuned LightGBM survival model with 5-fold CV |
| `run_deepsurv(X_imp, y_event, y_duration, label)` | Feature matrix, survival labels | CV C-index, fitted model, scaler | Optuna-tuned DeepSurv (pycox CoxPH) with early stopping |
| `calc_deepsurv_c(model, scaler, X, y_event, y_duration)` | Fitted model + scaler, test data | Test C-index, survival DataFrame | Evaluates DeepSurv on held-out test set |
| `weighted_ensemble(risk_score_dict, weights_dict, y_event, y_duration, label)` | Risk score and weight dicts, survival labels | Ensemble C-index, combined scores | Weighted average ensemble using C-index weights |
| `build_csf_imputer(df_baseline)` | Baseline DataFrame | Fitted LGB regressor, predictor cols | Trains CSF ABETA predictor from PET/MRI (Tier 3 imputation) |
| `save_checkpoint(name, obj)` | Name string, any object | — | Pickles object to `checkpoints/<name>.pkl` |
| `load_checkpoint(name)` | Name string | Unpickled object or None | Loads checkpoint if it exists |

### `postprocessing.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `calibration_plot(X_imp, y_event, y_duration, predict_proba_fn, horizon, model_name, cohort)` | Feature matrix, labels, prediction function, horizon | Saved calibration plot | Decile calibration plot at fixed time horizon |
| `km_risk_quartile(risk_scores, y_event, y_duration, model_name, cohort)` | Risk scores, survival labels | Saved KM plot | Kaplan-Meier curves stratified by risk quartile |
| `build_subject_time_matrix(df_all, rids, time_grid, features)` | Longitudinal DF, RID list, time grid, features | (n_subjects, n_timepoints, n_features) tensor + mask | Builds aligned longitudinal tensor for transformer input |

---

## Reproducibility Notes

- All random seeds are set via `RANDOM_SEED = 42` and passed explicitly to all models, CV splitters, and imputers.
- Model checkpoints are saved after each training run. Set `RETRAIN = False` to reproduce results from saved checkpoints without retraining.
- All figures are saved to `Baseline/figures/` automatically during notebook execution.
- ADNI data must not be committed to this repository per the ADNI Data Use Agreement. The `.gitignore` excludes all CSV files in `Baseline/tables/`.
