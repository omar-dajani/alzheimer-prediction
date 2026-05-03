# ADNI Exploratory Data Analysis (EDA)

This directory contains a collection of complementary exploratory data analyses (EDAs) performed on the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset.

Each EDA focuses on a different aspect of the data, including multimodal integration, feature-level behavior, MRI longitudinal structure, and statistical analysis. Together, they provide a comprehensive understanding of the dataset and inform downstream modeling decisions.

---

## Directory Structure

### 1. **Multimodal vs Single Modality**

**Purpose:**
Explores how different data modalities (genetics, plasma biomarkers, cognition, MRI-derived features, and imaging availability) interact and overlap.

**Key Focus Areas:**

* Dataset integration across modalities
* MRI vs tabular data availability
* Longitudinal structure (0–36 months)
* Cohort construction and feature selection

**Why it matters:**
Provides the **big-picture view** of the dataset and highlights limitations of multimodal modeling (e.g., sparse MRI coverage).

---

### 2. **Feature Deep Dive**

**Purpose:**
Performs a targeted, feature-level analysis of clinically relevant variables.

**Key Focus Areas:**

* Distributions of biomarkers, cognition, and MRI features
* Differences across CN, MCI, and AD groups
* Correlation analysis
* Longitudinal feature behavior

**Why it matters:**
Identifies **high-signal features** and validates known biological patterns in Alzheimer’s progression.

---

### 3. **MRI_Longitudinal_Dynamics**

**Purpose:**
Analyzes MRI scan availability and longitudinal patient participation patterns.

**Key Focus Areas:**

* Scan frequency per patient
* Follow-up duration and dropout behavior
* Differences between stable vs progressing patients
* Imaging vs tabular data gaps

**Why it matters:**
Reveals **critical biases in imaging data**, especially:

* high baseline-only participation
* better retention for progressing patients

---

### 4. **Statistical and Patient Trajectories**

**Purpose:**
Combines statistical testing with patient-level longitudinal visualization.

**Key Focus Areas:**

* Batch ANOVA for feature significance
* Feature distribution comparisons by diagnosis
* Individual patient trajectories over time
* Population-level data distribution

**Why it matters:**
Supports **feature selection and modeling readiness** by identifying statistically significant variables and understanding patient heterogeneity.

---

## How These EDAs Work Together

Each analysis answers a different question:

| EDA                                | Question Answered                                                        |
| ---------------------------------- | ------------------------------------------------------------------------ |
| Multimodal vs Single Modality      | *What data do we actually have, and how does it overlap?*                |
| Feature Deep Dive                  | *Which features matter and how do they behave?*                          |
| MRI Longitudinal Dynamics          | *How reliable and complete is MRI data over time?*                       |
| Statistical & Patient Trajectories | *Which features are statistically important and how do patients evolve?* |

Together, they provide:

* A full understanding of **data availability**
* Insight into **feature importance and behavior**
* Awareness of **longitudinal and modality biases**
* A strong foundation for **predictive modeling**

---

## Important Notes

* MRI data is **sparse relative to tabular data**, limiting multimodal modeling
* Many patients contribute only **baseline observations**
* MCI remains a **heterogeneous group**, complicating classification tasks
* Missingness varies significantly across modalities

---

## Next Steps

These EDAs inform downstream work such as:

* Feature engineering
* Predictive modeling (classification / survival analysis)
* Multimodal learning approaches
* Longitudinal modeling (time-series, sequence models)

---

## Summary

This directory represents a **multi-perspective exploration of the ADNI dataset**, combining:

* data integration
* feature analysis
* imaging dynamics
* statistical validation

to build a complete understanding of Alzheimer’s disease data structure and signal.
