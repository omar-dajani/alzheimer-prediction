"""
02_tabular_prep_improved_v2_aws.py — Tabular Integration for Improved MRI Pipeline v2 (AWS Version)

Identical processing to 02_tabular_prep_improved_v2.py but with Linux paths for AWS EC2.
Merges ADNIMERGE.csv with DXSUM diagnosis labels and MRI tensor paths pointing
to the improved skull-stripped tensors in data/processed/mri_tensors_improved_v2/.

Strict output requirements:
  - ALL original ADNIMERGE columns retained unmodified
  - Final row count == original ADNIMERGE row count (16,420)
  - Diagnosis mapped to numeric: CN -> 1.0, MCI -> 2.0, AD -> 3.0
  - MRI_File_Path points to data/processed/mri_tensors_improved_v2/{PTID}_{Visit}.npy
  - Front columns: PTID, VISCODE, Diagnosis, MRI_File_Path, AGE
  - Output: /home/ec2-user/ADNI_Project/data/processed/master_data_improved_04052026_v2.csv
"""

import os
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# Set to 1 for Tolerance Join (Recommended)
# Set to 2 for VISCODE Exact Match Join
# ==========================================
PREFERRED_METHOD = 2

# Paths pointing to the v2 pipeline outputs
MRI_TENSOR_SUBDIR = "data/processed/mri_tensors_improved_v2/"
MASTER_DATA_FILENAME = "master_data_improved_04052026_v2.csv"


def parse_adni_dates(date_series):
    """Safely converts standard dates ('2005-08-23') AND pesky Excel serials ('38603')."""
    s = date_series.astype(str).str.strip()
    is_excel = s.str.match(r'^\d+$') & (s != 'NaT') & (s != 'nan') & (s != 'None')

    parsed = pd.Series(pd.NaT, index=s.index)
    if (~is_excel).any():
        parsed.loc[~is_excel] = pd.to_datetime(s[~is_excel], errors='coerce')
    if is_excel.any():
        parsed.loc[is_excel] = pd.to_datetime(
            pd.to_numeric(s[is_excel]), origin='1899-12-30', unit='D'
        )
    return parsed


def extract_dxsum_labels(df):
    """
    Standardizes DXSUM diagnosis across all historical ADNI phases.
    Returns numeric labels: CN -> 1.0, MCI -> 2.0, AD -> 3.0.
    """
    df_out = df.copy()
    df_out['Diagnosis'] = np.nan

    # ADNI 3/4
    if 'DIAGNOSIS' in df_out.columns:
        diag = pd.to_numeric(df_out['DIAGNOSIS'], errors='coerce')
        df_out.loc[diag == 1, 'Diagnosis'] = 1.0  # CN
        df_out.loc[diag == 2, 'Diagnosis'] = 2.0  # MCI
        df_out.loc[diag == 3, 'Diagnosis'] = 3.0  # AD

    # ADNI GO/2 Tracking Codes
    if 'DXCHANGE' in df_out.columns:
        dxchange = pd.to_numeric(df_out['DXCHANGE'], errors='coerce')
        dx_map = {
            1: 1.0, 7: 1.0, 9: 1.0,   # CN
            2: 2.0, 4: 2.0, 8: 2.0,   # MCI
            3: 3.0, 5: 3.0, 6: 3.0,   # AD
        }
        df_out['Diagnosis'] = df_out['Diagnosis'].fillna(dxchange.map(dx_map))

    # ADNI 1
    if 'DXCURREN' in df_out.columns:
        dxcurren = pd.to_numeric(df_out['DXCURREN'], errors='coerce')
        dx_map = {1: 1.0, 2: 2.0, 3: 3.0}
        df_out['Diagnosis'] = df_out['Diagnosis'].fillna(dxcurren.map(dx_map))

    return df_out


def main():
    BASE_DIR = "/home/ec2-user/ADNI_Project"
    ADNIMERGE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'ADNIMERGE.csv')
    META_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'adni_all_mri_metadata.csv')
    DXSUM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'DXSUM_22Jan2026.csv')
    MASTER_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', MASTER_DATA_FILENAME)

    print("Loading Tabular datasets...")
    try:
        adni_merge = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
        mri_df = pd.read_csv(META_PATH)
        dxsum_df = pd.read_csv(DXSUM_PATH, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: {e}. Check file placement.")
        return

    original_row_count = len(adni_merge)
    print(f"ADNIMERGE rows: {original_row_count}")
    print(f"MRI metadata rows: {len(mri_df)}")
    print(f"DXSUM rows: {len(dxsum_df)}")

    # --- 0. Setup Unmodified Base Table (ADNIMERGE) ---
    adni_merge['original_index'] = np.arange(len(adni_merge))
    adni_merge['EXAMDATE_parsed'] = parse_adni_dates(adni_merge['EXAMDATE'])

    # --- 1. Clean MRI Metadata ---
    mri_df = mri_df.rename(columns={'Subject': 'PTID', 'Image Data ID': 'ImageID'})
    mri_df['Acq Date'] = parse_adni_dates(mri_df['Acq Date'])

    mri_df['Visit_clean'] = (
        mri_df['Visit']
        .astype(str)
        .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    )
    mri_df['MRI_File_Path'] = (
        MRI_TENSOR_SUBDIR + mri_df['PTID'] + "_" + mri_df['Visit_clean'] + ".npy"
    )

    # Check if v2 tensors exist on disk
    mri_df['Tensor_Exists'] = mri_df['MRI_File_Path'].apply(
        lambda x: os.path.exists(os.path.join(BASE_DIR, x))
    )
    existing_count = mri_df['Tensor_Exists'].sum()
    if existing_count > 0:
        mri_df = mri_df[mri_df['Tensor_Exists']].copy()
        print(f"Found {existing_count} v2 MRI tensors on disk.")
    else:
        print("WARNING: No v2 .npy files found. Run 01_mri_prep_improved_v2.py first!")
        print("Proceeding with all metadata paths for demonstration.")

    mri_clean = mri_df.dropna(subset=['Acq Date']).copy()

    # --- 2. Clean DXSUM Data ---
    dxsum_df['EXAMDATE_parsed'] = parse_adni_dates(dxsum_df['EXAMDATE'])
    dxsum_df = extract_dxsum_labels(dxsum_df)
    dxsum_clean = dxsum_df.dropna(subset=['EXAMDATE_parsed', 'Diagnosis']).copy()

    # =========================================================
    # METHOD 1: Left Join via 90-Day Tolerance
    # =========================================================
    valid_dates_mask = adni_merge['EXAMDATE_parsed'].notna()
    adni_base_sorted = adni_merge[valid_dates_mask].sort_values('EXAMDATE_parsed')
    adni_base_missing = adni_merge[~valid_dates_mask].copy()

    dxsum_sorted = dxsum_clean.sort_values('EXAMDATE_parsed')
    mri_sorted = mri_clean.sort_values('Acq Date')

    meth1 = pd.merge_asof(
        adni_base_sorted,
        dxsum_sorted[['PTID', 'EXAMDATE_parsed', 'Diagnosis']].rename(
            columns={'EXAMDATE_parsed': 'DX_DATE'}
        ),
        left_on='EXAMDATE_parsed',
        right_on='DX_DATE',
        by='PTID',
        direction='nearest',
        tolerance=pd.Timedelta(days=90),
    )

    meth1 = pd.merge_asof(
        meth1,
        mri_sorted[['PTID', 'Acq Date', 'MRI_File_Path']].rename(
            columns={'Acq Date': 'MRI_DATE'}
        ),
        left_on='EXAMDATE_parsed',
        right_on='MRI_DATE',
        by='PTID',
        direction='nearest',
        tolerance=pd.Timedelta(days=90),
    )

    meth1 = pd.concat([meth1, adni_base_missing], ignore_index=True)
    meth1 = meth1.sort_values('original_index').drop(
        columns=['DX_DATE', 'MRI_DATE'], errors='ignore'
    )

    # =========================================================
    # METHOD 2: Left Join via VISCODE Exact Match
    # =========================================================
    meth2_base = adni_merge.copy()
    meth2_base['Mapped_VISCODE'] = meth2_base['VISCODE'].astype(str).str.lower()

    dx_viscode_col = 'VISCODE2' if 'VISCODE2' in dxsum_clean.columns else 'VISCODE'
    dxsum_vis = dxsum_clean.copy()
    dxsum_vis['Mapped_VISCODE'] = (
        dxsum_vis[dx_viscode_col]
        .astype(str)
        .str.lower()
        .replace({'sc': 'bl', 'scmri': 'bl'})
    )
    dxsum_vis = dxsum_vis.sort_values('EXAMDATE_parsed').drop_duplicates(
        subset=['PTID', 'Mapped_VISCODE'], keep='last'
    )

    mri_vis = mri_clean.copy()
    mri_vis['Mapped_VISCODE'] = (
        mri_vis['Visit']
        .astype(str)
        .str.lower()
        .replace({'sc': 'bl', 'scmri': 'bl'})
    )
    mri_vis = mri_vis.sort_values('Acq Date').drop_duplicates(
        subset=['PTID', 'Mapped_VISCODE'], keep='last'
    )

    meth2 = pd.merge(
        meth2_base,
        dxsum_vis[['PTID', 'Mapped_VISCODE', 'Diagnosis']],
        on=['PTID', 'Mapped_VISCODE'],
        how='left',
    )

    meth2 = pd.merge(
        meth2,
        mri_vis[['PTID', 'Mapped_VISCODE', 'MRI_File_Path']],
        on=['PTID', 'Mapped_VISCODE'],
        how='left',
    )
    meth2 = meth2.sort_values('original_index').drop(
        columns=['Mapped_VISCODE'], errors='ignore'
    )

    # ---------------------------------------------------------
    # COMPARISON SUMMARY
    # ---------------------------------------------------------
    meth1_dx = meth1['Diagnosis'].notna().sum()
    meth1_mri = meth1['MRI_File_Path'].notna().sum()
    meth1_both = meth1[['Diagnosis', 'MRI_File_Path']].notna().all(axis=1).sum()
    meth2_dx = meth2['Diagnosis'].notna().sum()
    meth2_mri = meth2['MRI_File_Path'].notna().sum()
    meth2_both = meth2[['Diagnosis', 'MRI_File_Path']].notna().all(axis=1).sum()

    print("\n" + "=" * 65)
    print(" DATA AVAILABILITY COMPARISON (LEFT JOIN) — v2 PIPELINE")
    print("=" * 65)
    print(f"Total Canonical ADNIMERGE Base Rows Maintained: {original_row_count}")
    print(f"Method 1 Output Rows: {len(meth1)} (Match: {len(meth1) == original_row_count})")
    print(f"Method 2 Output Rows: {len(meth2)} (Match: {len(meth2) == original_row_count})\n")
    print("DIAGNOSIS COVERAGE:")
    print(f"  Method 1 (90-Day Tolerance): {meth1_dx}")
    print(f"  Method 2 (VISCODE String):   {meth2_dx}")
    print("\nMRI TENSOR COVERAGE:")
    print(f"  Method 1 (90-Day Tolerance): {meth1_mri}")
    print(f"  Method 2 (VISCODE String):   {meth2_mri}")
    print("\nRECORDS READY FOR DEEP LEARNING (Label + MRI Tensor):")
    print(f"  Method 1 (90-Day Tolerance): {meth1_both}")
    print(f"  Method 2 (VISCODE String):   {meth2_both}")
    print("=" * 65 + "\n")

    # Select preferred method
    if PREFERRED_METHOD == 1:
        print("Exporting Method 1 (Tolerance Join) for final Master Data v2...")
        final_df = meth1.copy()
    else:
        print("Exporting Method 2 (VISCODE Join) for final Master Data v2...")
        final_df = meth2.copy()

    # --- Format: retain ALL original ADNIMERGE columns unmodified ---
    original_adni_cols = [
        c for c in adni_merge.columns if c not in ['original_index', 'EXAMDATE_parsed']
    ]

    front_cols = ['PTID', 'VISCODE', 'Diagnosis', 'MRI_File_Path', 'AGE']
    front_cols = [c for c in front_cols if c in final_df.columns]

    other_cols = [c for c in original_adni_cols if c not in front_cols]
    master_data = final_df[front_cols + other_cols]

    # Verification: exact row-count match with ADNIMERGE
    assert len(master_data) == original_row_count, (
        f"Row mismatch! Output: {len(master_data)}, Input: {original_row_count}"
    )

    # Export (no index)
    os.makedirs(os.path.dirname(MASTER_DATA_PATH), exist_ok=True)
    master_data.to_csv(MASTER_DATA_PATH, index=False)

    print(f"SUCCESS! Master dataset exported to: {MASTER_DATA_PATH}")
    print(f"  Rows: {len(master_data)}")
    print(f"  Columns: {list(master_data.columns[:8])} ... ({len(master_data.columns)} total)")
    print(f"  Diagnosis distribution:")
    print(f"    {master_data['Diagnosis'].value_counts().sort_index().to_dict()}")
    print(f"  MRI paths populated: {master_data['MRI_File_Path'].notna().sum()}")
    print(f"  DL-ready (Dx + MRI): {master_data[['Diagnosis', 'MRI_File_Path']].notna().all(axis=1).sum()}")


if __name__ == "__main__":
    main()
