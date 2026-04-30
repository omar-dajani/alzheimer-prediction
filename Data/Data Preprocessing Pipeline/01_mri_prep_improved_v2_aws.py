"""
01_mri_prep_improved_v2_aws.py — Improved MRI Preprocessing Pipeline v2 (AWS Version)

Identical processing to 01_mri_prep_improved_v2.py but with Linux paths for AWS EC2.
Converts raw ADNI NIfTI files into deep-learning-ready 128x128x128 tensors.
Processes BOTH the original standardized NIfTIs AND the newly converted
DICOM-to-NIfTI files from Phase 2.

Pipeline per NIfTI (in strict order):
  1. Rigid registration to MNI152 template space
  2. deepbet skull stripping on the MNI-registered volume
  3. Tight 3D bounding-box crop of the brain mask
  4. Symmetric zero-pad to a perfect cube
  5. Isotropic resize to 128x128x128 via trilinear interpolation
  6. Robust percentile-clipped intensity normalization
  7. Save as float32 .npy to data/processed/mri_tensors_improved_v2/

Output directory: /home/ec2-user/ADNI_Project/data/processed/mri_tensors_improved_v2/
Output naming:   {PTID}_{Visit}.npy
"""

import os
import glob
import re
import tempfile
import shutil
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom

import ants
from deepbet import run_bet


# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = "/home/ec2-user/ADNI_Project"
RAW_ADNI_DIR = os.path.join(BASE_DIR, "data", "raw", "ADNI")
META_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "adni_all_mri_metadata.csv")
TENSOR_DIR = os.path.join(BASE_DIR, "data", "processed", "mri_tensors_improved_v2")
TARGET_SHAPE = (128, 128, 128)

# AWS: Auto-detect CPU count, leave headroom for deepbet GPU overhead
# Each worker uses ~2-4 GB RAM (ANTsPy + deepbet)
NUM_WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 14))


# ── Step 1: Rigid Registration to MNI152 ────────────────────────────────────
def register_to_mni(nifti_path: str) -> tuple:
    """
    Rigidly registers a raw NIfTI volume to the standard MNI152 1mm template
    using ANTsPy. Returns (registered nibabel Nifti1Image, tmpdir path).
    Caller is responsible for cleaning up tmpdir after the image is consumed.
    """
    mni_template = ants.image_read(ants.get_ants_data("mni"))
    moving = ants.image_read(nifti_path)

    registration = ants.registration(
        fixed=mni_template,
        moving=moving,
        type_of_transform="Rigid",
    )
    registered_ants = registration["warpedmovout"]

    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "registered_mni.nii.gz")
    ants.image_write(registered_ants, tmpfile)
    registered_nifti = nib.load(tmpfile)
    return registered_nifti, tmpdir


# ── Step 2: Skull Stripping with deepbet ─────────────────────────────────────
def skull_strip(registered_nifti: nib.Nifti1Image) -> np.ndarray:
    """
    Runs deepbet on an already-MNI-registered nibabel image.
    Returns the skull-stripped 3D numpy array.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(tmpdir, "registered.nii.gz")
        output_path = os.path.join(tmpdir, "stripped.nii.gz")
        mask_path = os.path.join(tmpdir, "mask.nii.gz")

        nib.save(registered_nifti, input_path)
        run_bet([input_path], [output_path], [mask_path])

        stripped_img = nib.load(output_path)
        stripped_data = stripped_img.get_fdata()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return stripped_data


# ── Step 3: Smart Bounding Box Crop ──────────────────────────────────────────
def bounding_box_crop(volume: np.ndarray) -> np.ndarray:
    """
    Finds the tightest 3D bounding box around all non-zero voxels
    and crops the volume to that box.
    """
    nonzero = np.argwhere(volume > 0)
    if nonzero.size == 0:
        return volume

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1

    return volume[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]


# ── Step 4: Symmetric Pad to a Perfect Cube ──────────────────────────────────
def pad_to_cube(volume: np.ndarray) -> np.ndarray:
    """
    Symmetrically zero-pads the shorter dimensions so the volume
    becomes a perfect cube.
    """
    shape = volume.shape
    max_dim = max(shape)

    pad_widths = []
    for s in shape:
        deficit = max_dim - s
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        pad_widths.append((pad_before, pad_after))

    return np.pad(volume, pad_widths, mode="constant", constant_values=0)


# ── Step 5: Isotropic Resize to 128^3 ─────────────────────────────────────────
def isotropic_resize(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resamples a perfect-cube volume down to target_shape using trilinear
    interpolation (scipy.ndimage.zoom, order=1).
    """
    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, factors, order=1)


# ── Step 6: Robust Percentile Intensity Normalization ─────────────────────────
def robust_normalize(volume: np.ndarray) -> np.ndarray:
    """
    1. Creates a foreground mask (voxels > 0).
    2. Computes the 1st and 99th percentile of intensities within the mask.
    3. Clips all voxels to that [p1, p99] range.
    4. Min-max normalizes the result to [0.0, 1.0].
    """
    foreground_mask = volume > 0
    if not foreground_mask.any():
        return volume

    foreground_values = volume[foreground_mask]
    p1 = np.percentile(foreground_values, 1)
    p99 = np.percentile(foreground_values, 99)

    volume_clipped = np.clip(volume, p1, p99)

    v_min = volume_clipped.min()
    v_max = volume_clipped.max()
    if v_max - v_min > 0:
        volume_norm = (volume_clipped - v_min) / (v_max - v_min)
    else:
        volume_norm = np.zeros_like(volume_clipped)

    return volume_norm


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def process_mri(nifti_path: str, target_shape: tuple = TARGET_SHAPE) -> np.ndarray:
    """
    Runs the complete 7-step pipeline on a single raw NIfTI file.
    Returns a float32 numpy array of shape target_shape.
    """
    registered_nifti, reg_tmpdir = register_to_mni(nifti_path)
    stripped = skull_strip(registered_nifti)
    shutil.rmtree(reg_tmpdir, ignore_errors=True)

    if stripped.ndim > 3:
        stripped = stripped[:, :, :, 0]

    cubed = pad_to_cube(stripped)
    resized = isotropic_resize(cubed, target_shape)
    normalized = robust_normalize(resized)

    return normalized.astype(np.float32)


def _process_one(task):
    """Worker function: process a single NIfTI and save the tensor."""
    nifti_path, out_path, image_id, out_filename = task
    t0 = time.time()
    try:
        tensor = process_mri(nifti_path)
        np.save(out_path, tensor)
        elapsed = time.time() - t0
        return image_id, out_filename, True, f"OK ({elapsed:.1f}s) shape={tensor.shape}"
    except Exception as e:
        elapsed = time.time() - t0
        return image_id, out_filename, False, f"FAILED ({elapsed:.1f}s): {e}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TENSOR_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  MRI Preprocessing Pipeline v2 (AWS) — NIfTI to 128^3 Tensors ({NUM_WORKERS} workers)")
    print("=" * 70)

    # Load metadata
    print("Loading metadata...")
    meta_df = pd.read_csv(META_CSV_PATH)
    meta_df = meta_df.rename(columns={"Image Data ID": "ImageID", "Subject": "PTID"})
    meta_df["ImageID"] = meta_df["ImageID"].astype(str).str.strip()

    meta_lookup = {}
    for _, row in meta_df.iterrows():
        iid = row["ImageID"]
        if iid.startswith("I"):
            meta_lookup[iid] = (row["PTID"], str(row["Visit"]))

    print(f"Metadata lookup: {len(meta_lookup)} Image IDs")

    # Discover all NIfTI files
    print("Searching for NIfTI files...")
    nii_files = glob.glob(os.path.join(RAW_ADNI_DIR, "**", "*.nii"), recursive=True)
    nii_gz_files = glob.glob(os.path.join(RAW_ADNI_DIR, "**", "*.nii.gz"), recursive=True)
    all_nii = nii_files + nii_gz_files
    print(f"Found {len(all_nii)} NIfTI files. Starting pipeline...\n")

    # Build task list
    tasks = []
    skipped = 0
    no_meta = 0
    seen_outputs = set()

    for nifti_path in all_nii:
        filename = os.path.basename(nifti_path)

        match = re.search(r"(I\d+)\.nii", filename, re.IGNORECASE)
        if not match:
            continue
        image_id = match.group(1)

        if image_id not in meta_lookup:
            no_meta += 1
            continue

        ptid, visit = meta_lookup[image_id]
        visit_clean = re.sub(r"[^a-zA-Z0-9_]", "", visit.replace(" ", "_"))
        out_filename = f"{ptid}_{visit_clean}.npy"
        out_path = os.path.join(TENSOR_DIR, out_filename)

        if os.path.exists(out_path) or out_path in seen_outputs:
            skipped += 1
            continue

        seen_outputs.add(out_path)
        tasks.append((nifti_path, out_path, image_id, out_filename))

    print(f"To process: {len(tasks)} | Skipped: {skipped} | No metadata: {no_meta}\n")

    # Process in parallel
    processed = 0
    errors = 0

    with Pool(NUM_WORKERS) as pool:
        for image_id, out_filename, success, msg in pool.imap_unordered(_process_one, tasks):
            if success:
                processed += 1
            else:
                errors += 1
            print(f"[{processed + errors}/{len(tasks)}] {image_id} -> {out_filename} {msg}")

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE v2 COMPLETE (AWS)")
    print(f"  Processed: {processed} | Skipped: {skipped} | Errors: {errors} | No metadata: {no_meta}")
    print(f"  Tensors saved to: {TENSOR_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
