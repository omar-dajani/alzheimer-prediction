# ADNI MPRAGE Longitudinal Data Analysis

## Key Insights from the Data Table

### 1. Exceptional Retention for "Progressors"

Patients whose cognitive health actively declines are monitored with incredible consistency, providing a dense, high-quality dataset for time-series modeling.

* **MCI ➔ AD (Late Decliners):** Of the 361 patients with at least one scan, **329 (91%)** have 4 or more sequential scans.
* **CN ➔ AD (Rapid Decliners):** Of the 41 patients with an initial scan, **39 (95%)** have 4 or more scans.

### 2. High Dropout Rates for "Stable" Patients

Conversely, healthy controls and stable patients drop out of the imaging study at much higher rates over time.

* **CN ➔ CN (Stable Healthy):** Plummets from 1,043 initial scans down to just **392** by the 4th scan (a **62% loss**).
* **MCI ➔ MCI (Stable Impaired):** Drops from 845 initial scans down to **421** by the 4th scan (a **50% loss**).

### 3. The Tabular vs. Imaging Data Gap

If you plan to build multimodal models (requiring *both* clinical tabular data and MRI data), beware of the missing tabular records in stable cohorts:

* **CN ➔ CN:** 1,043 patients have imaging data, but only **741** have tabular data. You lose over 300 healthy baseline patients immediately.
* **MCI ➔ AD:** Tabular tracking is near-perfect (395 tabular records for 399 diagnosed patients).