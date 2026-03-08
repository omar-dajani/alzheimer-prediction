# Alzheimer's Disease Progression — Multistate Survival Analysis
**Rice D2K Capstone · Spring 2026**  
**Professor:** Xinjie Lan
**Sponsor:** Cindy Zhang
**Mentor:** Antonio Mendoza Gonazales
**Team:**  Nathon Chavez, Omar Dajani, Eliza Iqbol, Savannah Nix, Fabrizio Pacheco, Evie Roth, & Shichen Tang 

---

## Overview

This project builds a multimodal survival analysis pipeline to predict Alzheimer's disease progression using longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Two cohorts are modeled:

- **MCI → Dementia** (primary): 958 subjects, median follow-up ~2.7 years
- **CN → Any Decline** (secondary): 824 subjects, median follow-up ~3.3 years, max ~17 years

The pipeline implements and compares six model families — Cox PH, Random Survival Forest, LightGBM, XGBoost AFT, DeepSurv, and DeepHit — plus a multistate Aalen-Johansen model for disease state-occupation probabilities across the full CN → MCI → Dementia trajectory.

---

## Results Summary

### MCI → Dementia (C-index, 5-fold CV)

| Model | C-index |
|-------|---------|
| XGBoost AFT | **0.908** |
| LightGBM Survival | 0.907 |
| Domain Ensemble (OOF) | 0.902 |
| Random Survival Forest | 0.879 |
| DeepSurv | 0.873 |
| Cox PH | 0.839 ± 0.015 |
| DeepHit | 0.838 |
| Multistate Cox | 0.826 |

### CN → Decline (C-index, 5-fold CV)

| Model | C-index |
|-------|---------|
| XGBoost AFT | **0.884** |
| Random Survival Forest | 0.846 |
| LightGBM Survival | 0.835 |
| DeepSurv | 0.817 |
| DeepHit | 0.767 |
| Multistate Cox (CN→MCI) | 0.768 |
| Cox PH | 0.753 ± 0.056 |

---

## Repository Structure

```
alzheimer-prediction/
├── ADNI_Survival_Pipeline.ipynb   # Main pipeline (Colab notebook)
├── EDA/
│   └── eda_feature_deepdive.py    # Exploratory data analysis
├── LightGBM/
│   └── __init__.py
├── Transformer/
│   └── __init__.py                # Placeholder for future LSTM/Transformer work
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data

This project uses the **ADNI Merged Dataset (ADNIMERGE)**, accessed via the Alzheimer's Disease Neuroimaging Initiative data portal at [adni.loni.usc.edu](https://adni.loni.usc.edu).

> **The data file is not included in this repository.** Access requires registration and approval through the ADNI data use agreement. Redistribution is prohibited under ADNI terms.

To reproduce results, download `ADNIMERGE.csv` from the ADNI portal and place it at:
```
/content/drive/MyDrive/ADNI_capstone/ADNIMERGE_28Feb2026.csv
```

---

## Pipeline Overview

### Preprocessing
- DX harmonization: SMC→CN, EMCI/LMCI→MCI, AD→Dementia
- MCI→CN reversion removal (sponsor-specified)
- ComBat harmonization audit for MRI scanner batch effects (1.5T vs 3T)

### Imputation (3-tier)
1. Longitudinal nearest-neighbor fill (±1 year window)
2. MICE / IterativeImputer for remaining gaps
3. Two-stage CSF imputation: LightGBM predicts ABETA from PET + MRI (R²=0.414)

### Feature Engineering
- ICV-normalized MRI volumes (Hippocampus, Entorhinal, Ventricles)
- Amyloid composite (ABETA CSF + AV45 PET)
- Neurodegeneration composite (TAU + PTAU + FDG)
- Cognitive composites (ability + severity)
- Pre-cutoff OLS slopes per feature (leakage-free — strictly pre-event visits only)
- Slope velocity (acceleration) features
- APOE4 interaction terms
- Slope concordance (simultaneous cognitive + structural decline)

### Models
| Cell | Model | Notes |
|------|-------|-------|
| 14 | Cox PH | Baseline, PH assumption tested |
| 15 | Random Survival Forest | 40-trial Optuna tuning |
| 16 | LightGBM + XGBoost AFT | GPU-accelerated, SHAP analysis |
| 17 | DeepSurv | Neural Cox PH via pycox |
| 18 | DeepHit | Discrete-time survival |
| 19 | Multistate Cox + Aalen-Johansen | CN→MCI→Dementia state-occupation probabilities |
| 20 | Ensemble | Weighted + domain expert stacking |
| 21–23 | Calibration, KM quartiles, comparison table | |
| 24 | Longitudinal tensor | (958 subjects, 12 timepoints, 14 features) for future RNN/Transformer |

---

## Key Findings

- **Cognitive trajectory slopes dominate prediction**: `slope_LDELTOTAL` (Logical Memory) and `slope_CDRSB` are the top features across RSF and LightGBM importance — rate of decline matters more than snapshot severity for MCI→Dementia
- **XGBoost AFT outperforms all models on CN→Decline** (0.884 vs 0.846 RSF): the AFT objective handles the long right tail of CN follow-up (up to 17 years) better than ranking-based objectives
- **PH violations** for `slope_LDELTOTAL` (p=0.0005) and `slope_Hippocampus` (p=0.028) motivate the non-parametric models
- **Direct CN→Dementia conversion is rare** (5 events / 824 subjects): almost all subjects transition through MCI first, validating the three-state model
- **Domain ensemble meta-learner**: cognitive domain weight 0.888, imaging 0.431, CSF/PET −0.134 — biomarker signal is largely subsumed by cognitive trajectories once all domains are combined

---

## Requirements

```
lifelines
scikit-survival
pycox
lightgbm
xgboost
optuna
shap
scikit-learn
neuroCombat
torchtuples
torch
numpy
pandas
matplotlib
seaborn
scipy
```

Install via:
```bash
pip install lifelines scikit-survival pycox lightgbm xgboost optuna shap \
            neuroCombat torchtuples --quiet
```

---

## Reproducing Results

1. Mount Google Drive and place the ADNI CSV at the path above
2. Open `ADNI_Survival_Pipeline.ipynb` in Google Colab (Pro recommended, T4 GPU)
3. Run Cell 1 to install dependencies, then restart runtime
4. Run all cells sequentially (Cells 2–24)
5. Outputs saved to `/content/drive/MyDrive/ADNI_capstone/outputs/`

Total runtime: approximately 2–3 hours on T4 GPU (dominated by RSF Optuna tuning ~21 min, DeepSurv ~3 min, DeepHit ~3 min)

---

## Citation

Data used in preparation of this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.

> Petersen RC, et al. Alzheimer's Disease Neuroimaging Initiative (ADNI): Clinical characterization. *Neurology*. 2010;74(3):201-209.
