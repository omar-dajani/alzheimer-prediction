# ADNI Multimodal Alzheimer’s Disease EDA

This repository contains a consolidated exploratory data analysis workflow for the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset. The analysis integrates clinical diagnosis, APOE genetics, plasma biomarkers, cognitive scores, MRI-derived brain volume features, and raw MRI/DICOM folder coverage.

## Project Goal

The goal is to create one clean, reproducible EDA notebook that replaces multiple fragmented notebooks and documents the full data preparation and exploratory analysis pipeline.

The final notebook is:

```text
notebooks/adni_multimodal_eda_FULL.ipynb
```

## Data Sources

The workflow uses ADNI data files including:

- `DXSUM.csv` — longitudinal diagnosis records
- `ADSP_COGN_SCORE.csv` — cognitive score summaries
- `PLASMA.csv` — plasma biomarkers
- `APOERES.csv` — APOE genotype data
- `ADSP_BIOMARKER_SCORE.csv` — MRI-derived brain volume features
- `ADNIAlpha/` and `ADNIBeta/` — raw MRI/DICOM image folders

## Workflow Summary

The notebook is organized into the following sections:

1. Setup and path definitions
2. Raw tabular dataset inspection
3. MRI/DICOM image inventory
4. Optional raw MRI scan QA example
5. Cohort construction, merge, and initial imputation
6. Longitudinal cleanup and feature selection
7. Descriptive EDA by diagnosis
8. Multimodal coverage and longitudinal trends
9. Final EDA takeaways

## Final Modeling Dataset

The cleaned modeling dataset is generated from the merged intermediate data and restricted to a 0–36 month longitudinal window.

Expected final dataset summary:

| Metric | Value |
|---|---:|
| Observations | 11,631 |
| Patients | 3,762 |
| Final Features | 25 |
| Month Range | 0–36 |
| Missing Values | 0 |

## Modality Coverage

| Modality Group | Patients |
|---|---:|
| MRI + Tabular | 581 |
| Tabular Only | 3,181 |
| MRI Only | 0 |

MRI coverage is useful but sparse compared with the tabular cohort, so downstream modeling should account for modality missingness and potential selection bias.

## Key EDA Findings

- APOE genotype and plasma biomarkers provide strong disease-state signal.
- p-Tau217 and amyloid-related biomarkers differ across CN, MCI, and AD groups.
- Cognitive scores decline across the expected clinical severity gradient.
- MRI-derived hippocampal, entorhinal, and ventricle measures support expected neurodegeneration patterns.
- MCI is heterogeneous and may require special modeling attention.
- Raw MRI folders are readable and can be inventoried, but direct imaging modeling is left as future work.

## Important Implementation Notes

- Diagnosis, cognition, plasma, and MRI-derived features are merged by `RID` and `VISCODE`.
- APOE genotype is static and merged by `RID`.
- Plasma and cognition are initially imputed by diagnosis-group median.
- MRI imputation is deferred until after key MRI features are selected.
- Final output is saved as:

```text
data/02_intermediate/adni_final_for_modeling.csv
```

## Limitations

- MRI data is available for a minority of patients.
- Plasma biomarker missingness may affect downstream predictive performance.
- Diagnosis-group median imputation preserves class structure but can introduce optimistic separation if not handled carefully in predictive modeling.
- Raw MRI images are inspected for availability but not used directly for modeling in this notebook.

## Future Work

- Add predictive modeling with train/test split discipline.
- Compare XGBoost, ElasticNet, survival models, and longitudinal models.
- Add SHAP or permutation importance for model interpretation.
- Incorporate direct MRI image-based features or deep learning models.
- Improve imputation using training-only pipelines to avoid leakage.
