import kagglehub

print("=" * 60)
print("  ADNI Preprocessed Tabular Dataset: Kaggle Download")
print("=" * 60)

# 1. Master CSV v3 (~10 MB)
print("\n Downloading Master CSV...")
csv_path = kagglehub.dataset_download("fabriziopacheco/adni-master-csv")
print(f"  Done: {csv_path}")
