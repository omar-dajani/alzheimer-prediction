# ADNI Merge: Feature Deep Dive EDA

Exploratory data analysis notebook for the Rice University D2K Capstone project (Spring 2026), sponsored by Cindy Zhang. This notebook investigates the ADNI longitudinal dataset to characterize features relevant to Alzheimer's disease progression modeling, with a focus on building intuition ahead of multistate survival analysis.

---

## Overview

**Dataset:** `ADNIMERGE_28Feb2026.csv` — 16,421 visit records across 2,430 unique subjects from all ADNI phases (ADNI1, ADNIGO, ADNI2, ADNI3).

**Goal:** Understand the structure, missingness, discriminative power, and longitudinal behavior of key biomarkers before feeding them into a survival modeling pipeline.

**Diagnosis groups:** All analyses use three harmonized labels — `CN` (Cognitively Normal), `MCI` (Mild Cognitive Impairment), and `Dementia`.

---

## Feature Groups Analyzed

| Category | Features |
|---|---|
| Demographics | Age (`AGE`), Sex (`PTGENDER`), Education (`PTEDUCAT`) |
| Genetic | APOE4 allele count (`APOE4`) |
| Cognitive | MMSE, CDR-SB, ADAS-11, ADAS-13, Logical Memory (`LDELTOTAL`), RAVLT, FAQ |
| CSF Biomarkers | Amyloid-β (`ABETA`), Total Tau (`TAU`), Phospho-Tau (`PTAU`) |
| MRI Volumetric | Hippocampus, Entorhinal, Ventricles, WholeBrain, Fusiform, MidTemp (all normalized by ICV) |

---

## Notebook Structure

The notebook produces 20 figures, each preceded by a markdown cell documenting its purpose and key observations. Figures are saved to `eda_figures/`.

### Figure Index

| # | File | Description |
|---|---|---|
| 01 | `01_overview.png` | Visit distribution, baseline diagnosis breakdown, sex by DX |
| 02 | `02_missingness.png` | Missing data rates by feature, color-coded by severity |
| 03 | `03_demographics.png` | Age histograms, education boxplots, APOE4 allele counts by DX |
| 04 | `04_cognitive_scores.png` | Notched boxplots of 6 cognitive tests by baseline DX |
| 05 | `05_csf_biomarkers.png` | CSF Aβ42, Total Tau, Phospho-Tau distributions by DX |
| 06 | `06_mri_volumes.png` | ICV-normalized MRI volumes (Hippocampus, Entorhinal, Ventricles, etc.) by DX |
| 07 | `07_correlation_heatmap.png` | Spearman correlation heatmap across all key features at baseline |
| 08 | `08_longitudinal_trajectories.png` | Mean trajectories over 8 years for MMSE, Hippocampus, CDR-SB, CSF Aβ |
| 09 | `09_mci_conversion.png` | Baseline feature distributions: MCI converters vs. stable MCI |
| 10 | `10_feature_discriminability.png` | Per-feature AUROC for CN vs. Dementia discrimination |
| 11a | `11a_cn_mci_overview.png` | CN subject outcome breakdown and converter vs. stable CN violins |
| 11b | *(embedded in 11a)* | Stable CN vs. CN→MCI baseline feature violins |
| 11c | *(embedded in 11a)* | AUROC comparison: CN→MCI vs. MCI→Dementia prediction tasks |
| 12 | `12_coverage.png` | Visit counts by ADNI phase and maximum follow-up duration by DX |
| 13 | `13_slopes_by_dx.png` | Annualized OLS feature slopes (trajectories) by DX group |
| 13b/c | `13b_slopes_mci_conversion.png`, `13c_slopes_cn_mci_conversion.png` | Slope distributions for MCI converters and CN→MCI converters |
| 14a/b | `14a_km_mci_dementia.png`, `14b_km_cn_mci.png` | Kaplan-Meier survival curves, stratified by APOE4 and CSF amyloid status |
| 15a/b | `15a_modality_heatmap.png`, `15b_modality_over_time.png` | Baseline data availability per subject × modality; modality coverage across visits |
| 16a/b | `16a_apoe4_interaction.png`, `16b_apoe4_heatmap.png` | APOE4 × biomarker interactions within each DX group |
| 17 | `17_batch_effects.png` | Scanner/protocol batch effects across ADNI phases |
| 18 | `18_cn_cognitive_stability.png` | MMSE trajectories and slope distributions for baseline-CN subjects |
| 19a/b | `19a_completeness_heatmap.png`, `19b_completeness_lines.png` | Feature completeness across visit codes (BL through M120) |
| 20 | `20_pet_csf_correlation.png` | Cross-modality scatter plots: PET vs. CSF biomarkers |

---

## Key Findings

**Missingness is structured, not random.** Demographic features are near-complete; cognitive scores are ~28–31% missing; MRI volumetrics are ~46–49% missing longitudinally (but only ~10–22% at baseline); CSF biomarkers exceed 50% missing dataset-wide, reflecting the selective nature of lumbar puncture consent.

**Strongest discriminators of CN vs. Dementia** (by AUROC at baseline): CDR-SB (~1.00), FAQ (~0.99), ADAS-13, Logical Memory, and Hippocampal volume (~0.91). Age and education are weak predictors (AUROC ~0.61–0.63) and serve mainly as covariates.

**APOE4 is the most important genetic feature.** The proportion of subjects with ≥1 risk allele scales strongly with diagnosis severity. This gradient is confirmed by the APOE4 × biomarker interaction panels.

**MCI-to-Dementia conversion is predictable.** Even at baseline, converters show lower hippocampal volume, lower Aβ42, higher p-Tau, and worse cognitive scores compared to stable MCI subjects.

**Batch effects exist across ADNI phases.** Hippocampal volume shows measurable protocol-driven jumps at the ADNI1→ADNI2 transition, motivating the ComBat harmonization step in the main pipeline.

**TAU and PTAU are nearly redundant** (Spearman r ≈ 0.98), providing strong justification for using one as a predictor or combining them.

---

## Requirements

```bash
pip install pandas matplotlib seaborn scipy scikit-learn --break-system-packages
```

**Data:** Place `ADNIMERGE_28Feb2026.csv` in the same directory as the notebook. This file is governed by the ADNI Data Use Agreement and **must not** be committed to version control.

---

## Data Access

ADNI data is available to approved researchers at [adni.loni.usc.edu](https://adni.loni.usc.edu). Access requires an approved application through the ADNI Data Sharing and Publications Committee.

---
