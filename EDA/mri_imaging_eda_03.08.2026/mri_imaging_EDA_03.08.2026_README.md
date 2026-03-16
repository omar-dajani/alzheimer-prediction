# MPRAGE MRI Sessions & Patient Follow-Up Analysis

## Executive Summary

This analysis covers deduplicated MPRAGE MRI scanning sessions for **2,922 patients**. The data is heavily right-skewed, revealing high initial dropout, a solid core of short-term participants, and a rare, highly valuable group of long-term patients.

## Key Metrics

* **Total Patients:** 2,922
* **Total Unique Scans:** 10,868
* **Scans per Patient:** Average 3.72 | Median 3 | Max 16
* **Follow-Up Duration:** Average 2.65 years | Max 19.98 years

## Primary Insights

* **High Early Attrition (~38%):** Over a third of the cohort (**1,116 patients**) completed only a single baseline scan and have exactly 0.0 years of follow-up.
* **The "Typical" Patient Journey:** For those who continue past baseline, participation clusters heavily around the **1 to 3-year** marks. The middle 50% of the cohort completes between **1 and 5 total scans**.
* **Long-Term Outliers:** A small, dedicated group of "super-participants" forms a long right tail in the data. They have been monitored continuously for up to **20 years**, receiving up to **16 unique scans**.

## Strategic Takeaway

When building models, researchers must account for the massive volume of cross-sectional, baseline-only data. However, the remaining longitudinal data is robust, typically spanning **2 to 3 years**, with rare cases offering nearly two decades of continuous disease tracking.