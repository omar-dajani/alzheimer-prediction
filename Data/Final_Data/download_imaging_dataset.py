import kagglehub

print("=" * 60)
print("  ADNI Preprocessed Dataset v2 — Kaggle Download")
print("  2 datasets: MRI Tensors + Flow Tensors")
print("=" * 60)

# 1. MRI Tensors — 9,072 files, 128x128x128 float32 (~71 GB)
print("\n[2/3] Downloading MRI Tensors (~71 GB, please wait)...")
mri_path = kagglehub.dataset_download("fabriziopacheco/adni-mri-tensors")
print(f"  Done: {mri_path}")

# 2. Flow Tensors — 4,123 files, 3x128x128x128 float32 (~97 GB)
print("\n[3/3] Downloading Flow Tensors (~97 GB, please wait)...")
flow_path = kagglehub.dataset_download("fabriziopacheco/adni-flow-tensors")
print(f"  Done: {flow_path}")

print("\n" + "=" * 60)
print("  All downloads complete!")
print(f"  MRI Tensors:  {mri_path}")
print(f"  Flow Tensors: {flow_path}")
print("=" * 60)
