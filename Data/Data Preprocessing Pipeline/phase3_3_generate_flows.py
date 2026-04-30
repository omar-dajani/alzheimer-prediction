"""
phase3_3_generate_flows.py — Phase 3.3: Longitudinal Deformation Field Generation (AWS Version)

Computes SyN-based deformation fields between consecutive MRI visits for each
patient to capture disease velocity (structural change over time). Produces
annualized 3D vector flow tensors compatible with the LongFormer architecture.

Pipeline per patient (chronological visit order):
  1. Baseline visit: Save a zero-filled (3, 128, 128, 128) array ("Missing Flow")
  2. Follow-up visits: Run SyNOnly registration (prior -> current), extract the
     forward warp field, transpose to (3, 128, 128, 128), divide by time delta
     in years to annualize disease velocity, and save.
  3. Update the master CSV with a new Flow_File_Path column.

Input:  /home/ec2-user/ADNI_Project/data/processed/master_data_improved_04052026_v2.csv
        /home/ec2-user/ADNI_Project/data/processed/mri_tensors_improved_v2/*.npy
Output: /home/ec2-user/ADNI_Project/data/processed/flow_tensors_v2/{PTID}_{VISCODE}_flow.npy
        /home/ec2-user/ADNI_Project/data/processed/master_data_improved_04052026_v3.csv

Dependencies: antspyx, numpy, pandas, tqdm
"""

import os
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd
import ants

from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = "/home/ec2-user/ADNI_Project"
MASTER_CSV_V2 = os.path.join(BASE_DIR, "data", "processed", "master_data_improved_04052026_v2.csv")
MASTER_CSV_V3 = os.path.join(BASE_DIR, "data", "processed", "master_data_improved_04052026_v3.csv")
MRI_TENSOR_DIR = os.path.join(BASE_DIR, "data", "processed", "mri_tensors_improved_v2")
FLOW_TENSOR_DIR = os.path.join(BASE_DIR, "data", "processed", "flow_tensors_v2")
FLOW_TENSOR_SUBDIR = "data/processed/flow_tensors_v2/"

VOLUME_SHAPE = (128, 128, 128)
FLOW_SHAPE = (3, 128, 128, 128)
MIN_DELTA_YEARS = 0.5  # Safety clamp: prevent gradient explosion from re-scans

# SyN registration is CPU-heavy (~2-4 GB RAM per worker); cap workers
NUM_WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 10))


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tensor_as_ants(npy_path: str) -> ants.ANTsImage:
    """Load a (128, 128, 128) float32 .npy tensor and convert to ANTs image."""
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim > 3:
        arr = arr[:, :, :, 0]
    return ants.from_numpy(arr)


def compute_flow(current_path: str, prior_path: str) -> np.ndarray:
    """
    Compute the forward deformation field from prior scan to current scan
    using SyNOnly (images are already rigidly aligned from Phase 3.1).

    Returns a (3, 128, 128, 128) float32 numpy array.
    Cleans up temporary .nii.gz / .mat files produced by ANTs.
    """
    current_img = load_tensor_as_ants(current_path)
    prior_img = load_tensor_as_ants(prior_path)

    res = ants.registration(
        fixed=current_img,
        moving=prior_img,
        type_of_transform="SyNOnly",
    )

    # Read the forward warp field (first entry is the .nii.gz warp)
    fwd_warp_path = res["fwdtransforms"][0]
    warp_field = ants.image_read(fwd_warp_path)
    warp_array = warp_field.numpy()  # (128, 128, 128, 3)

    # Transpose: channel-last -> channel-first => (3, 128, 128, 128)
    flow = np.transpose(warp_array, (3, 0, 1, 2)).astype(np.float32)

    # Clean up ALL temporary transform files to prevent disk fill-up
    for path_list in [res["fwdtransforms"], res["invtransforms"]]:
        for fpath in path_list:
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except OSError:
                pass

    return flow


def process_patient(patient_visits: list) -> list:
    """
    Process all longitudinal visits for a single patient.

    Args:
        patient_visits: List of dicts, each with keys:
            PTID, VISCODE, mri_abs_path, examdate, flow_out_path, flow_rel_path

    Returns:
        List of (PTID, VISCODE, flow_rel_path, success, message) tuples.
    """
    results = []
    ptid = patient_visits[0]["PTID"]

    for i, visit in enumerate(patient_visits):
        out_path = visit["flow_out_path"]
        rel_path = visit["flow_rel_path"]
        viscode = visit["VISCODE"]

        # Skip logic: already computed
        if os.path.exists(out_path):
            results.append((ptid, viscode, rel_path, True, "SKIPPED (exists)"))
            continue

        t0 = time.time()

        try:
            if i == 0:
                # Baseline: zero-filled "Missing Flow"
                flow = np.zeros(FLOW_SHAPE, dtype=np.float32)
                np.save(out_path, flow)
                elapsed = time.time() - t0
                results.append((ptid, viscode, rel_path, True, f"OK baseline zeros ({elapsed:.1f}s)"))
            else:
                # Follow-up: compute deformation from prior to current
                current_path = visit["mri_abs_path"]
                prior_path = patient_visits[i - 1]["mri_abs_path"]

                flow = compute_flow(current_path, prior_path)

                # Time-scaling: annualize the deformation velocity
                current_date = visit["examdate"]
                prior_date = patient_visits[i - 1]["examdate"]

                if pd.notna(current_date) and pd.notna(prior_date):
                    delta_years = (current_date - prior_date).days / 365.25
                else:
                    # Fallback: use AGE difference if dates are missing
                    current_age = visit.get("age")
                    prior_age = patient_visits[i - 1].get("age")
                    if pd.notna(current_age) and pd.notna(prior_age) and current_age != prior_age:
                        delta_years = float(current_age) - float(prior_age)
                    else:
                        delta_years = 1.0  # Default to 1 year if no time info

                safe_delta = max(delta_years, MIN_DELTA_YEARS)
                flow = flow / safe_delta

                np.save(out_path, flow.astype(np.float32))
                elapsed = time.time() - t0
                results.append((
                    ptid, viscode, rel_path, True,
                    f"OK ({elapsed:.1f}s) delta={delta_years:.2f}y safe={safe_delta:.2f}y"
                ))

        except Exception as e:
            elapsed = time.time() - t0
            results.append((ptid, viscode, rel_path, False, f"FAILED ({elapsed:.1f}s): {e}"))

    return results


def _worker(patient_visits: list) -> list:
    """Top-level worker function for multiprocessing (must be picklable)."""
    return process_patient(patient_visits)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FLOW_TENSOR_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  Phase 3.3: Longitudinal Deformation Fields ({NUM_WORKERS} workers)")
    print("=" * 70)

    # ── Load master CSV ──────────────────────────────────────────────────
    print("Loading master CSV v2...")
    df = pd.read_csv(MASTER_CSV_V2, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Parse EXAMDATE
    df["EXAMDATE_parsed"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    # Filter to rows with a valid MRI tensor on disk
    has_mri = df["MRI_File_Path"].notna()
    df_mri = df[has_mri].copy()

    df_mri["mri_abs_path"] = df_mri["MRI_File_Path"].apply(
        lambda p: os.path.join(BASE_DIR, p)
    )
    df_mri["tensor_exists"] = df_mri["mri_abs_path"].apply(os.path.exists)
    df_mri = df_mri[df_mri["tensor_exists"]].copy()
    print(f"  Rows with existing MRI tensor: {len(df_mri)}")

    if len(df_mri) == 0:
        print("ERROR: No MRI tensors found on disk. Run Phase 3.1 first.")
        return

    # ── Group by patient and sort chronologically ────────────────────────
    print("Grouping by patient and sorting chronologically...")

    patient_tasks = []

    for ptid, group in df_mri.groupby("PTID"):
        # Sort by EXAMDATE first; fallback to AGE if all dates are missing
        if group["EXAMDATE_parsed"].notna().any():
            group = group.sort_values("EXAMDATE_parsed")
        elif "AGE" in group.columns and group["AGE"].notna().any():
            group = group.sort_values("AGE")

        visits = []
        for _, row in group.iterrows():
            viscode = str(row["VISCODE"]) if pd.notna(row.get("VISCODE")) else "unknown"
            viscode_clean = viscode.replace(" ", "_")

            flow_filename = f"{ptid}_{viscode_clean}_flow.npy"
            flow_out_path = os.path.join(FLOW_TENSOR_DIR, flow_filename)
            flow_rel_path = FLOW_TENSOR_SUBDIR + flow_filename

            visits.append({
                "PTID": ptid,
                "VISCODE": viscode,
                "mri_abs_path": row["mri_abs_path"],
                "examdate": row["EXAMDATE_parsed"],
                "age": row.get("AGE"),
                "flow_out_path": flow_out_path,
                "flow_rel_path": flow_rel_path,
            })

        if visits:
            patient_tasks.append(visits)

    total_visits = sum(len(v) for v in patient_tasks)
    print(f"  Patients: {len(patient_tasks)} | Total visits to process: {total_visits}\n")

    # ── Process patients in parallel ─────────────────────────────────────
    processed = 0
    errors = 0
    skipped = 0

    # Collect all (PTID, VISCODE) -> flow_rel_path mappings for CSV update
    flow_map = {}

    with Pool(NUM_WORKERS) as pool:
        with tqdm(total=total_visits, desc="Computing deformation fields") as pbar:
            for patient_results in pool.imap_unordered(_worker, patient_tasks):
                for ptid, viscode, flow_rel_path, success, msg in patient_results:
                    if success:
                        if "SKIPPED" in msg:
                            skipped += 1
                        else:
                            processed += 1
                        flow_map[(ptid, viscode)] = flow_rel_path
                    else:
                        errors += 1
                        print(f"  ERROR: {ptid}/{viscode}: {msg}")
                    pbar.update(1)

    print(f"\n{'=' * 70}")
    print(f"  DEFORMATION FIELD GENERATION COMPLETE")
    print(f"  Computed: {processed} | Skipped: {skipped} | Errors: {errors}")
    print(f"  Flow tensors saved to: {FLOW_TENSOR_DIR}")
    print(f"{'=' * 70}\n")

    # ── Update master CSV with Flow_File_Path ────────────────────────────
    print("Updating master CSV with Flow_File_Path column...")

    # Reload the FULL master CSV (not just the MRI-filtered subset)
    master_df = pd.read_csv(MASTER_CSV_V2, low_memory=False)

    # Map (PTID, VISCODE) -> flow_rel_path
    master_df["Flow_File_Path"] = master_df.apply(
        lambda row: flow_map.get(
            (row["PTID"], str(row["VISCODE"]) if pd.notna(row.get("VISCODE")) else "unknown"),
            np.nan,
        ),
        axis=1,
    )

    # Verify the flow files actually exist on disk
    def verify_flow(path):
        if pd.isna(path):
            return np.nan
        abs_path = os.path.join(BASE_DIR, path)
        return path if os.path.exists(abs_path) else np.nan

    master_df["Flow_File_Path"] = master_df["Flow_File_Path"].apply(verify_flow)

    # Insert Flow_File_Path right after MRI_File_Path
    cols = list(master_df.columns)
    cols.remove("Flow_File_Path")
    mri_idx = cols.index("MRI_File_Path") if "MRI_File_Path" in cols else 3
    cols.insert(mri_idx + 1, "Flow_File_Path")
    master_df = master_df[cols]

    # Save v3
    master_df.to_csv(MASTER_CSV_V3, index=False)

    flow_count = master_df["Flow_File_Path"].notna().sum()
    dl_ready = master_df[
        ["Diagnosis", "MRI_File_Path", "Flow_File_Path"]
    ].notna().all(axis=1).sum()

    print(f"SUCCESS! Master dataset v3 exported to: {MASTER_CSV_V3}")
    print(f"  Rows: {len(master_df)}")
    print(f"  Flow paths populated: {flow_count}")
    print(f"  DL-ready (Dx + MRI + Flow): {dl_ready}")
    print(f"  Columns: {list(master_df.columns[:7])} ... ({len(master_df.columns)} total)")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
