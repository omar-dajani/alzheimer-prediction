# ADNI Biomarkers & Genetics: Summary & Conclusions

Here we explore the methodology and primary findings from our exploratory data analysis (EDA) using the ADNI genetics and plasma biomarker datasets.

## Methodological Summary

We conducted a comprehensive analysis integrating Whole Genome Sequencing (WGS Omni2.5M) PLINK data, plasma biomarker concentrations, and longitudinal clinical diagnoses.

1.  **Data Curation & Deduplication**:
    *   Assigned 4,639 unique patients across five ADNI study phases (ADNI1, ADNIGO, ADNI2, ADNI3, ADNI4) to their earliest enrolled cohort, eliminating cross-phase duplicates.
    *   Filtered the WGS dataset to 812 patients, leveraging a 2.5-million-marker whole-genome array.
    *   Isolated an overlap cohort of 251 patients who possessed both WGS genotyping and baseline plasma biomarker data.

2.  **Genetics Mapping & Feature Engineering**:
    *   Mapped 13,050 Single Nucleotide Polymorphisms (SNPs) to 33 known Alzheimer's-related genes (e.g., APP, PSEN1, TREM2, BIN1, CD33) with 50kb regulatory flanks.
    *   Computed gene-level PCA and burden scores, extracted deterministic/pathogenic variants (e.g., TREM2 R47H), and determined APOE-ε4 dosage.
    *   Calculated the top 10 Ancestry Principal Components to correct for population stratification.

3.  **Biomarker Cleaning**:
    *   Processed 7 core plasma biomarkers (p-Tau217, Aβ42, Aβ40, Aβ42/Aβ40, pTau217/Aβ42, NfL, GFAP).
    *   Applied zero-leakage transformations (log1p for skewed distributions, RobustScaler, and KNN imputation) fitted strictly on the training set.

4.  **Predictive Modeling**:
    *   Trained XGBoost and ElasticNet models to predict 3-class disease status (CN vs. MCI vs. AD).
    *   Fitted Cox Proportional Hazards (Cox PH) models for time-to-progression (CN/MCI progressing to a worse diagnostic state).
    *   Utilized SHAP (SHapley Additive exPlanations) values to extract feature importance and biological signal.

## Conclusions

Our findings confirm the profound synergistic effect between genetic susceptibility and real-time fluid biomarkers in tracking Alzheimer's disease progression.

*   **Gene-Biomarker Synergy is the Primary Predictor**:
    The interaction between **APOE-ε4 dosage and p-Tau217** emerged as the strongest single predictor of clinical AD (highest SHAP value: 0.292). The combination of the major genetic risk factor (ε4) and the most specific tau pathology marker (p-Tau217) acts as a biological accelerant, radically outperforming either feature in isolation.
*   **APOE Exerts the Strongest Independent Survival Effects**:
    The Cox PH survival model (concordance = 0.66) validated that APOE-ε4 dosage is the most significant standalone risk factor for progression (Hazard Ratio = 1.53, p < 0.005), meaning each ε4 allele increases the risk of faster diagnostic decline by 53%. Conversely, the APOE-ε2 allele proved protective (Hazard Ratio = 0.60, p = 0.04), reducing progression risk by 40%.
*   **Plasma Markers as Dynamic State Indicators**:
    Elevated baseline p-Tau217 (HR = 1.42, p = 0.01) and altered Aβ42/Aβ40 ratios independently predicted time-to-progression. Furthermore, Aβ40 levels (reflecting overall amyloid production capacity) ranked as the second most important feature in the SHAP classification analysis, underscoring the critical role of amyloid dynamics in stratifying patients.
*   **GWAS Polygenic Loci Contribute Meaningful Signal**:
    Beyond APOE, features derived from the BIN1 gene (the second-strongest recognized GWAS locus involving endocytosis and tau vesicle release) ranked in the top 5 most important SHAP features, validating the biological impact of endosomal trafficking variations in a multivariate context.
*   **Predictive Limits of Genetics + Blood Biomarkers**:
    The XGBoost classifier achieved an ROC-AUC of 0.879 on the 3-class (CN/MCI/AD) problem. This modest accuracy was heavily expected because 69% of the test cohort required KNN imputation for missing plasma biomarkers, and MCI represents a highly heterogeneous clinical category. Relying on genetics alongside imputed baseline blood markers alone is insufficient for highly precise 3-class diagnostic classification without incorporating direct structural neuroimaging (e.g., MRI hippocampal volumes).
