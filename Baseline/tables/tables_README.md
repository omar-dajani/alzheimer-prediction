# Data

The ADNI data files are not included in this repository due to the ADNI Data Use Agreement.
You must apply for access independently and place the data here before running the pipeline.

## Applying for Access

Visit [adni.loni.usc.edu](https://adni.loni.usc.edu) and submit a data access application.
Access is typically granted within a few days for academic researchers.

## Data Preparation

**[PLACEHOLDER — to be completed by data acquisition lead]**

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
