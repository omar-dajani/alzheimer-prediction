The dataset used in this project was created using the ADNIMERGE package in RStudio, which contains curated ADNI datasets covering all study phases up to ADNI4. Within the R package, the data are distributed across multiple datasets (tables) rather than a single unified file.
To construct the dataset used in this project, the relevant tables were first loaded in RStudio and exported as CSV files. Because the datasets originate from multiple ADNI phases and package tables, visit codes were standardized across all datasets prior to merging. This ensured that visits from different phases corresponded to a consistent naming scheme.

After standardizing the visit codes, the tables were merged using the following identifiers:
`RID` – unique participant identifier
`VISCODE` – standardized visit code representing the visit timepoint

Standardizing these identifiers allowed records from different package datasets to be aligned correctly for the same participant and visit. The merged dataset therefore contains longitudinal participant information, including demographics, diagnostic labels, and MRI-derived measurements.

The following tables were used in this pipeline and exported as CSV files:<br>
`adrs.csv` – cognitive assessment scores (ADAS-related measures)<br>
`biomarkers.csv` – biomarker measurements<br>
`subjects.csv` – participant demographic and subject-level information<br>
`UCSFFSX7.csv` – structural MRI measurements derived from FreeSurfer segmentation