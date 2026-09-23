from pathlib import Path
from collections import Counter

from PIL import Image


# --------------------------------------------------
# 1. Dataset path
# --------------------------------------------------

dataset_path = Path("data/raw/crop_health")


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
# 3. Check dataset path
# --------------------------------------------------

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {dataset_path}"
    )


# --------------------------------------------------
# 4. Find class folders
# --------------------------------------------------

class_folders = sorted(
    [
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    ]
)


print("Number of classes:", len(class_folders))

print("\nClasses:")
for folder in class_folders:
    print("-", folder.name)


# --------------------------------------------------
# 5. Count images in each class
# --------------------------------------------------

total_images = 0
image_formats = Counter()
image_dimensions = Counter()
corrupted_images = []


print("\n========== Images Per Class ==========")

for class_folder in class_folders:

    images = [
        file
        for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    print(f"{class_folder.name}: {len(images)}")

    total_images += len(images)

    # ----------------------------------------------
    # Check every image
    # ----------------------------------------------

    for image_path in images:

        image_formats[image_path.suffix.lower()] += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            # Reopen after verify() to read dimensions
            with Image.open(image_path) as image:
                image_dimensions[image.size] += 1

        except Exception:
            corrupted_images.append(str(image_path))


# --------------------------------------------------
# 6. Dataset summary
# --------------------------------------------------

print("\n========== Dataset Summary ==========")

print("Total images:", total_images)

print("\nImage formats:")
for extension, count in image_formats.items():
    print(f"{extension}: {count}")

print("\nImage dimensions:")
for dimension, count in image_dimensions.most_common():
    print(f"{dimension}: {count}")


# --------------------------------------------------
# 7. Corrupted image check
# --------------------------------------------------

print("\n========== Corrupted Images ==========")

print("Corrupted images:", len(corrupted_images))

if corrupted_images:
    print("\nCorrupted image paths:")

    for image_path in corrupted_images:
        print("-", image_path)


# --------------------------------------------------
# 8. Final status
# --------------------------------------------------

print("\n========================================")
print("CROP HEALTH DATASET INSPECTION COMPLETE")
print("========================================")