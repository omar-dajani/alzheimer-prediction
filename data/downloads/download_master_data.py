import kagglehub

print("Connecting to Kaggle to download the dataset...")
print("Please wait, this is ~27GB and may take some time.")

# Download latest version
path = kagglehub.dataset_download("fabriziopacheco/adni-master-data")

print("Download Complete!")
print("Path to your mri_tensors and master_data.csv:", path)