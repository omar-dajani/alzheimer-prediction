# ADNI Feature Deep Dive EDA

This exploratory analysis focuses on a targeted set of clinical, cognitive, biomarker, and imaging features relevant to Alzheimer's disease progression.

Unlike broader pipeline-focused EDAs, this notebook performs a **feature-level deep dive**, examining distributions, missingness, correlations, and longitudinal behavior across diagnosis groups.

---

## Overview

The analysis investigates how key features differ across:

- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Dementia (AD)

It emphasizes **interpretability and clinical signal** rather than data engineering or modeling pipelines.

---

## Dataset Summary

- Observations: ~16,421
- Patients: ~2,430
- Longitudinal coverage: up to ~8 years
- Focused feature subset from sponsor specification

---

## Features Analyzed

### Demographics
- Age
- Sex
- Education

### Genetics
- APOE4 status

### Cognitive Scores
- MMSE
- CDR-SB
- Additional neuropsychological tests

### CSF Biomarkers
- Amyloid-β (ABETA)
- Total Tau (TAU)
- Phosphorylated Tau (PTAU)

### MRI Features
- Hippocampal volume
- Other brain regions (normalized by intracranial volume)

---

## Analysis Sections

### 1. Dataset Overview
- Distribution of visit counts per patient
- Diagnosis breakdown
- Longitudinal structure

### 2. Missingness Analysis
- Feature-level missing data ranking
- Identification of high-risk variables

### 3. Demographic & Genetic Distributions
- Age distributions by diagnosis
- APOE4 enrichment in disease groups

### 4. Cognitive Score Comparisons
- Boxplots across CN / MCI / AD
- Clear separation of disease severity

### 5. CSF Biomarker Analysis
- Amyloid and tau differences by diagnosis
- Biological signal validation

### 6. MRI Feature Analysis
- Volume differences across diagnosis
- Normalization by intracranial volume

### 7. Correlation Analysis
- Spearman correlation heatmap
- Relationships between biomarkers, cognition, and MRI

### 8. Longitudinal Trajectories
- Feature evolution over time
- Disease progression trends

### 9. MCI Conversion Analysis
- Baseline differences between:
  - MCI converters → AD
  - MCI non-converters
- Early predictive signal identification

---

## Key Insights

- Cognitive scores strongly differentiate CN, MCI, and AD
- APOE4 is enriched in disease populations
- CSF biomarkers show expected Alzheimer’s pathology patterns:
  - ↓ Amyloid-β
  - ↑ Tau
- MRI features reflect structural neurodegeneration
- Longitudinal trends reinforce progressive decline patterns
- MCI conversion analysis highlights early indicators of disease progression

---

## Limitations

- Missingness in biomarkers may bias comparisons
- Analysis is primarily descriptive (no predictive modeling)
- Baseline-focused comparisons may not capture full disease dynamics
- Feature set is limited to sponsor-selected variables

---

## Role in Overall Project

This EDA complements other analyses by:

- Providing **deep understanding of key features**
- Validating **biological and clinical signal**
- Identifying **features most relevant for modeling**
- Supporting downstream feature selection and modeling decisions

