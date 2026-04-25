import kagglehub
import os
os.environ['KAGGLE_USERNAME'] = "iinarixf0x"
os.environ['KAGGLE_KEY'] = "KGAT_2d81ea1701506605d237619894afa5db"
print("Connecting to Kaggle to download the skull-stripped dataset...")
print("Please wait, this may take some time depending on your connection.")

# Download latest version
path = kagglehub.dataset_download("fabriziopacheco/adni_master_data_skustr")

print("Download Complete!")
print("Path to your skull-stripped mri_tensors and master_data.csv:", path)
