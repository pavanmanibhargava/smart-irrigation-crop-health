from src.data.crop_health_dataset import (
    CropHealthDataset,
    train_transform,
)


# --------------------------------------------------
# Dataset path
# --------------------------------------------------

train_csv = "data/processed/crop_health/train.csv"


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

dataset = CropHealthDataset(
    csv_file=train_csv,
    transform=train_transform,
)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("Number of images:", len(dataset))

print("\nClass mapping:")
print(dataset.class_to_index)


# --------------------------------------------------
# Load one image
# --------------------------------------------------

image, label = dataset[0]


# --------------------------------------------------
# Check loaded image
# --------------------------------------------------

print("\nLoaded image successfully.")

print("Image shape:", image.shape)
print("Label:", label)
print("Image data type:", image.dtype)