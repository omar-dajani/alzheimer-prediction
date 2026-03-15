# Data

The ADNI data files are not included in this repository due to the ADNI Data Use Agreement.
You must apply for access independently and place the data here before running the pipeline.

## Applying for Access

Visit [adni.loni.usc.edu](https://adni.loni.usc.edu) and submit a data access application.
Access is typically granted within a few days for academic researchers.

## Data Preparation

The dataset used in this project was created using the ADNIMERGE package in RStudio, which contains curated ADNI datasets covering all study phases up to ADNI4. Within the R package, the data are distributed across multiple datasets (tables) rather than a single unified file. To construct the dataset used in this project, the relevant tables were first loaded in RStudio and exported as CSV files. Because the datasets originate from multiple ADNI phases and package tables, visit codes were standardized across all datasets prior to merging. This ensured that visits from different phases corresponded to a consistent naming scheme.

After standardizing the visit codes, the tables were merged using the following identifiers: RID – unique participant identifier VISCODE – standardized visit code representing the visit timepoint

Standardizing these identifiers allowed records from different package datasets to be aligned correctly for the same participant and visit. The merged dataset therefore contains longitudinal participant information, including demographics, diagnostic labels, and MRI-derived measurements.

The following tables were used in this pipeline and exported as CSV files:
adrs.csv – cognitive assessment scores (ADAS-related measures)
biomarkers.csv – biomarker measurements
subjects.csv – participant demographic and subject-level information
UCSFFSX7.csv – structural MRI measurements derived from FreeSurfer segmentation

Describe here:
- Which ADNI tables were downloaded
- How the ADNIMERGE and ADNI4 data were merged into a single CSV
- Any preprocessing steps applied before the file was saved here

## Expected File

Once access is granted and the data is prepared, place the merged CSV file in this directory:

```
Baseline/tables/ADNIMERGE_08Mar2026.csv
```

The notebook will fail with a clear error message if this file is not present:

```python
assert DATA_PATH.exists(), f"CSV not found at {DATA_PATH} -- place your ADNIMERGE CSV in Baseline/tables/"
```

## Files in this directory

The following output files are generated automatically by the pipeline and saved here:

| File | Description |
|---|---|
| `slopes_mci.csv` | Per-subject longitudinal OLS slopes for the MCI cohort |
| `slopes_cn.csv` | Per-subject longitudinal OLS slopes for the CN cohort |
| `master.csv` | Assembled baseline feature matrix after all preprocessing |
| `subjects.csv` | Subject-level metadata and survival labels |
| `biomarkers.csv` | CSF and PET biomarker summary per subject |
| `adrs.csv` | ADRS composite score table |
