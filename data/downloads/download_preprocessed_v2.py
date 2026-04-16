import kagglehub

print("=" * 60)
print("  ADNI Preprocessed Dataset v2 — Kaggle Download")
print("  3 datasets: Master CSV + MRI Tensors + Flow Tensors")
print("=" * 60)

# 1. Master CSV v3 (~10 MB)
print("\n[1/3] Downloading Master CSV v3...")
csv_path = kagglehub.dataset_download("fabriziopacheco/adni-master-csv")
print(f"  Done: {csv_path}")

# 2. MRI Tensors — 9,072 files, 128x128x128 float32 (~71 GB)
print("\n[2/3] Downloading MRI Tensors (~71 GB, please wait)...")
mri_path = kagglehub.dataset_download("fabriziopacheco/adni-mri-tensors")
print(f"  Done: {mri_path}")

# 3. Flow Tensors — 4,123 files, 3x128x128x128 float32 (~97 GB)
print("\n[3/3] Downloading Flow Tensors (~97 GB, please wait)...")
flow_path = kagglehub.dataset_download("fabriziopacheco/adni-flow-tensors")
print(f"  Done: {flow_path}")

print("\n" + "=" * 60)
print("  All downloads complete!")
print(f"  Master CSV:   {csv_path}")
print(f"  MRI Tensors:  {mri_path}")
print(f"  Flow Tensors: {flow_path}")
print("=" * 60)
