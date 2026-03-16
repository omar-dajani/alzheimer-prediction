# Documentation for `patient_count_distribution_EDA.ipynb`

## Overview
This notebook performs exploratory data analysis (EDA) on patient count distributions. It is designed to analyze and visualize data from specific tables related to patient information.

## Prerequisites

- **Python Environment:** Ensure you have a Python environment with Jupyter Notebook installed.
- **Install Required Libraries:** Install them using `pip install pandas numpy matplotlib seaborn` if needed.

## Required Data

Before running the notebook, ensure the following tables are available in the `Tables` directory (located in the same directory as the notebook):

- `All_Subjects_All_Images_20Feb2026.csv`
- `All_Subjects_DXSUM_20Feb2026.csv`
- `All_Subjects_BIOMARK_20Feb2026.csv`
- `All_Subjects_DTIROI_MEAN_20Feb2026.csv`
- `All_Subjects_DTIROI_ROBUSTMEAN_20Feb2026.csv`
- `All_Subjects_FOXLABBSI_20Feb2026.csv`
- `All_Subjects_UCSFFSX7_20Feb2026.csv`

> **Note:** The exact table names and formats required are specified in the notebook's data loading cells. Please refer to those cells for precise filenames and schema requirements.

## How to Run

1. **Place Tables:** Copy the required CSV files into the `Tables` directory.
2. **Open Notebook:** Launch Jupyter Notebook and open `patient_count_distribution_EDA.ipynb`.
3. **Run Cells:** Execute each cell in order to perform the EDA. The notebook will load data from the `Tables` directory and generate visualizations and summary statistics.

## Output

The notebook will produce plots and tables summarizing the distribution of patient counts, helping you understand trends and patterns in the data.