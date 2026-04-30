"""
phase2_dicom_to_nifti_aws.py — DICOM-to-NIfTI Conversion Pipeline (AWS Version)

Identical processing to phase2_dicom_to_nifti.py but with Linux paths for AWS EC2.
Converts raw DICOM series into "Standardized" NIfTI files that approximate
ADNI's 5-step baseline preprocessing:
  1. MPR (Multi-Planar Reconstruction) via dcm2niix
  2. GradWarp Proxy via rigid/affine registration (ANTsPy)
  3. B1 & N3 Correction via N4 Bias Field Correction (ANTsPy)
  4. Scaled (Intensity Normalization) via histogram matching to Gold Standard
  5. Save mimicking ADNI nested folder structure

Input:  /home/ec2-user/ADNI_Project/data/raw/ADNI_03.22.2026/ADNI/{PTID}/{Desc}/{Date}/{ImageID}/*.dcm
Output: /home/ec2-user/ADNI_Project/data/raw/ADNI/{PTID}/MPR__GradWarp__B1_Correction__N3__Scaled/{Date}/{ImageID}/...nii
"""

import os
import re
import glob
import json
import subprocess
import tempfile
import shutil
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import nibabel as nib
import ants
from skimage.exposure import match_histograms

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = "/home/ec2-user/ADNI_Project"
DICOM_ROOT = os.path.join(BASE_DIR, "data", "raw", "ADNI_03.22.2026", "ADNI")
NIFTI_ROOT = os.path.join(BASE_DIR, "data", "raw", "ADNI")
GOLD_HIST_PATH = os.path.join(BASE_DIR, "data", "processed", "gold_standard_histogram.npy")
REPORT_PATH = os.path.join(BASE_DIR, "data", "processed", "phase1_structure_report.json")
LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "phase2_conversion_log.json")

# ADNI standardized processing type name
PROCESSING_TYPE = "MPR__GradWarp__B1_Correction__N3__Scaled"

# AWS: Use more workers — adjust based on instance vCPU count
# c6i.4xlarge = 16 vCPU -> 14 workers
# c6i.24xlarge = 96 vCPU -> 80 workers
import multiprocessing
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)


def discover_dicom_series(dicom_root):
    """
    Discover all DICOM series directories.
    Structure: {PTID}/{Description}/{Date}/{ImageID}/*.dcm
    Returns list of dicts with series metadata.
    """
    series_list = []
    subjects = sorted(os.listdir(dicom_root))

    for ptid in subjects:
        ptid_dir = os.path.join(dicom_root, ptid)
        if not os.path.isdir(ptid_dir):
            continue

        for desc in os.listdir(ptid_dir):
            desc_dir = os.path.join(ptid_dir, desc)
            if not os.path.isdir(desc_dir):
                continue

            for date_str in os.listdir(desc_dir):
                date_dir = os.path.join(desc_dir, date_str)
                if not os.path.isdir(date_dir):
                    continue

                for image_id in os.listdir(date_dir):
                    series_dir = os.path.join(date_dir, image_id)
                    if not os.path.isdir(series_dir):
                        continue

                    # Verify DICOM files exist
                    dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
                    if len(dcm_files) == 0:
                        continue

                    series_list.append({
                        "ptid": ptid,
                        "description": desc,
                        "date": date_str,
                        "image_id": image_id,
                        "dicom_dir": series_dir,
                        "num_slices": len(dcm_files),
                    })

    return series_list


def build_output_path(ptid, date_str, image_id):
    """
    Build the ADNI-mimicking output path for a converted NIfTI.
    Returns (output_dir, output_filename).
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + "000"
    series_id = "S00000"

    output_dir = os.path.join(
        NIFTI_ROOT, ptid, PROCESSING_TYPE, date_str, image_id
    )
    output_filename = (
        f"ADNI_{ptid}_MR_{PROCESSING_TYPE}_Br_{timestamp}_{series_id}_{image_id}.nii"
    )
    return output_dir, output_filename


def check_already_converted(ptid, image_id):
    """Check if this series was already converted to NIfTI."""
    pattern = os.path.join(NIFTI_ROOT, ptid, "**", image_id, "*.nii")
    existing = glob.glob(pattern, recursive=True)
    return len(existing) > 0


def step1_dcm2niix(dicom_dir, tmpdir):
    """
    Step 1: MPR — Convert DICOM to NIfTI using dcm2niix.
    Returns path to the output NIfTI file.
    """
    cmd = [
        "dcm2niix",
        "-z", "n",
        "-f", "output",
        "-o", tmpdir,
        "-b", "n",
        dicom_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    nii_files = glob.glob(os.path.join(tmpdir, "*.nii"))
    if not nii_files:
        raise FileNotFoundError(f"dcm2niix produced no NIfTI output in {tmpdir}")

    if len(nii_files) > 1:
        nii_files.sort(key=lambda f: os.path.getsize(f), reverse=True)

    return nii_files[0]


def step2_gradwarp_proxy(nifti_path):
    """
    Step 2: GradWarp Proxy — Affine registration to MNI template.
    """
    mni_template = ants.image_read(ants.get_ants_data("mni"))
    moving = ants.image_read(nifti_path)

    registration = ants.registration(
        fixed=mni_template,
        moving=moving,
        type_of_transform="Affine",
    )

    corrected = registration["warpedmovout"]
    return corrected


def step3_n4_bias_correction(ants_image):
    """
    Step 3: B1 & N3 Correction — N4 Bias Field Correction.
    """
    corrected = ants.n4_bias_field_correction(ants_image)
    return corrected


def step4_histogram_matching(ants_image, gold_histogram):
    """
    Step 4: Scaled — Intensity normalization via histogram matching.
    """
    data = ants_image.numpy()

    n_bins = len(gold_histogram)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fg_mask = data > 0
    if not fg_mask.any():
        return ants_image

    dmin, dmax = data[fg_mask].min(), data[fg_mask].max()
    if dmax - dmin == 0:
        return ants_image

    data_norm = np.zeros_like(data)
    data_norm[fg_mask] = (data[fg_mask] - dmin) / (dmax - dmin)

    gold_cdf = np.cumsum(gold_histogram)
    gold_cdf = gold_cdf / gold_cdf[-1]

    n_fg = fg_mask.sum()
    ref_quantiles = np.linspace(0, 1, n_fg)
    ref_values = np.interp(ref_quantiles, gold_cdf, bin_centers)

    fg_values = data_norm[fg_mask]
    sort_idx = np.argsort(fg_values)
    matched = np.zeros_like(fg_values)
    matched[sort_idx] = np.sort(ref_values)

    result = np.zeros_like(data)
    result[fg_mask] = matched * (dmax - dmin) + dmin

    return ants_image.new_image_like(result)


def step5_save_nifti(ants_image, output_dir, output_filename):
    """
    Step 5: Save the processed NIfTI mimicking ADNI structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    ants.image_write(ants_image, output_path)
    return output_path


def process_single_series(series_info, gold_histogram):
    """Run the full 5-step pipeline on a single DICOM series."""
    ptid = series_info["ptid"]
    image_id = series_info["image_id"]
    date_str = series_info["date"]
    dicom_dir = series_info["dicom_dir"]

    tmpdir = tempfile.mkdtemp()
    try:
        nifti_path = step1_dcm2niix(dicom_dir, tmpdir)
        corrected = step2_gradwarp_proxy(nifti_path)
        bias_corrected = step3_n4_bias_correction(corrected)
        intensity_matched = step4_histogram_matching(bias_corrected, gold_histogram)
        output_dir, output_filename = build_output_path(ptid, date_str, image_id)
        output_path = step5_save_nifti(intensity_matched, output_dir, output_filename)

        return output_path, None
    except Exception as e:
        return None, str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_gold_histogram = None


def _init_worker(gold_hist):
    global _gold_histogram
    _gold_histogram = gold_hist


def _process_one(series_info):
    t0 = time.time()
    output_path, error = process_single_series(series_info, _gold_histogram)
    elapsed = time.time() - t0
    return series_info["ptid"], series_info["image_id"], output_path, error, round(elapsed, 1)


def main():
    print("=" * 70)
    print(f"  PHASE 2: DICOM-to-NIfTI Conversion Pipeline (AWS — {NUM_WORKERS} workers)")
    print("=" * 70)

    # Load Gold Standard histogram
    if not os.path.exists(GOLD_HIST_PATH):
        print("ERROR: Gold Standard histogram not found. Run phase1 first!")
        return
    gold_histogram = np.load(GOLD_HIST_PATH)
    print(f"Loaded Gold Standard histogram ({len(gold_histogram)} bins)")

    # Discover all DICOM series
    print("\nDiscovering DICOM series...")
    all_series = discover_dicom_series(DICOM_ROOT)
    print(f"Found {len(all_series)} DICOM series from {len(set(s['ptid'] for s in all_series))} subjects")

    # Filter out already-converted series
    todo = []
    skipped = 0
    for s in all_series:
        if check_already_converted(s["ptid"], s["image_id"]):
            skipped += 1
        else:
            todo.append(s)

    print(f"Already converted: {skipped}")
    print(f"To-do list: {len(todo)} series\n")

    if len(todo) == 0:
        print("Nothing to convert!")
        return

    # Process in parallel
    results = {"converted": [], "errors": []}
    completed = 0

    with Pool(NUM_WORKERS, initializer=_init_worker, initargs=(gold_histogram,)) as pool:
        for ptid, image_id, output_path, error, elapsed in pool.imap_unordered(_process_one, todo):
            completed += 1
            if output_path:
                print(f"[{completed}/{len(todo)}] {ptid} / {image_id} OK ({elapsed}s)")
                results["converted"].append({
                    "ptid": ptid,
                    "image_id": image_id,
                    "output_path": output_path,
                    "time_sec": elapsed,
                })
            else:
                print(f"[{completed}/{len(todo)}] {ptid} / {image_id} FAILED: {error}")
                results["errors"].append({
                    "ptid": ptid,
                    "image_id": image_id,
                    "error": error,
                })

    # Save log
    with open(LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 COMPLETE")
    print(f"  Converted: {len(results['converted'])} | Errors: {len(results['errors'])}")
    print(f"  Log saved to: {LOG_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
