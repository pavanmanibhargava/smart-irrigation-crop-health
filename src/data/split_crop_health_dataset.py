from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

dataset_path = Path("data/raw/crop_health")

output_path = Path("data/processed/crop_health")

output_path.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Supported image formats
# --------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# 3. Collect image paths and labels
# --------------------------------------------------

data = []

for class_folder in sorted(dataset_path.iterdir()):

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    for image_path in class_folder.iterdir():

        if (
            image_path.is_file()
            and image_path.suffix.lower() in image_extensions
        ):
            data.append(
                {
                    "image_path": str(image_path),
                    "class": class_name,
                }
            )


df = pd.DataFrame(data)


# --------------------------------------------------
# 4. Show dataset information
# --------------------------------------------------

print("Total images:", len(df))

print("\nClass distribution:")
print(df["class"].value_counts())


# --------------------------------------------------
# 5. Create training and temporary sets
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"],
)


# --------------------------------------------------
# 6. Split temporary set into validation and test
# --------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"],
)


# --------------------------------------------------
# 7. Reset indexes
# --------------------------------------------------

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# --------------------------------------------------
# 8. Save split files
# --------------------------------------------------

train_df.to_csv(
    output_path / "train.csv",
    index=False,
)

validation_df.to_csv(
    output_path / "validation.csv",
    index=False,
)

test_df.to_csv(
    output_path / "test.csv",
    index=False,
)


# --------------------------------------------------
# 9. Display split sizes
# --------------------------------------------------

print("\n========== Dataset Split ==========")

print("Training images:", len(train_df))
print("Validation images:", len(validation_df))
print("Test images:", len(test_df))


# --------------------------------------------------
# 10. Display class distribution
# --------------------------------------------------

print("\nTraining distribution:")
print(train_df["class"].value_counts())

print("\nValidation distribution:")
print(validation_df["class"].value_counts())

print("\nTest distribution:")
print(test_df["class"].value_counts())


# --------------------------------------------------
# 11. Final status
# --------------------------------------------------

print("\n========================================")
print("CROP HEALTH DATASET SPLIT COMPLETE")
print("========================================")