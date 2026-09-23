"""
Crop Health Model Training Script
==================================
Trains a pretrained ResNet18 to classify tomato crop health
into 10 disease/healthy classes.

Strategy — FAST BASELINE (transfer learning):
  • Load pretrained ResNet18 (ImageNet weights).
  • FREEZE all feature-extraction layers (~11 M params).
  • Replace and train ONLY the final FC layer (~5 K params).
  • 1 epoch, Adam, lr=0.001.

Uses the existing CropHealthDataset and transforms from
src/data/crop_health_dataset.py.

Usage:
    python -m src.models.train_crop_health_model
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data.crop_health_dataset import (
    CropHealthDataset,
    train_transform,
    validation_transform,
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

train_csv = "data/processed/crop_health/train.csv"
validation_csv = "data/processed/crop_health/validation.csv"

model_save_path = Path("models/crop_health_resnet18.pth")

# Make sure the output directory exists
model_save_path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Hyperparameters
# --------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1        # Fast baseline — single epoch
NUM_WORKERS = 0       # 0 for reliable Windows CPU training
NUM_CLASSES = 10


# --------------------------------------------------
# 3. Create datasets using the existing loader
# --------------------------------------------------

print("Loading datasets...")

train_dataset = CropHealthDataset(
    csv_file=train_csv,
    transform=train_transform,
)

# Pass the same class_to_index mapping from train to validation
# so that the label encoding is consistent across splits
validation_dataset = CropHealthDataset(
    csv_file=validation_csv,
    transform=validation_transform,
    class_to_index=train_dataset.class_to_index,
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class mapping:     {train_dataset.class_to_index}")


# --------------------------------------------------
# 4. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# --------------------------------------------------
# 5. Set up device (CPU)
# --------------------------------------------------

device = torch.device("cpu")
print(f"Device: {device}")


# --------------------------------------------------
# 6. Build the model  (freeze backbone, train FC only)
# --------------------------------------------------

# Load pretrained ResNet18 with default ImageNet weights.
# If the weights are cached locally they will be reused;
# if downloading fails (no internet) we exit with a clear message.
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception as exc:
    print(
        "\n[ERROR] Failed to load pretrained ResNet18 weights.\n"
        "If you have no internet connection, ensure the weights are "
        "cached in ~/.cache/torch/hub/checkpoints/.\n"
        f"Original error: {exc}"
    )
    sys.exit(1)

# FREEZE all feature-extraction layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully-connected layer to output 10 classes.
# Original fc: Linear(in_features=512, out_features=1000)
# This new layer's parameters are *unfrozen* by default.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

# Quick sanity check — only fc params should be trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResNet18 (pretrained, backbone frozen)")
print(f"Trainable params: {trainable_params:,} / {total_params:,}")


# --------------------------------------------------
# 7. Loss function and optimizer (train FC only)
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# --------------------------------------------------
# 8. Training loop — 1 epoch
# --------------------------------------------------

print(f"\n========== Training Started (1 epoch) ==========\n")

start_time = time.time()

# --- Training phase ---
model.train()

running_train_loss = 0.0
correct_train = 0
total_train = 0

for batch_index, (images, labels) in enumerate(train_loader):

    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track loss and accuracy
    running_train_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs, 1)
    correct_train += (predicted == labels).sum().item()
    total_train += labels.size(0)

    # Print progress every 50 batches
    if (batch_index + 1) % 50 == 0:
        print(
            f"  Batch {batch_index + 1}/{len(train_loader)} | "
            f"Running Acc: {100.0 * correct_train / total_train:.2f}%"
        )

train_loss = running_train_loss / total_train
train_accuracy = 100.0 * correct_train / total_train

# --- Validation phase ---
model.eval()

running_val_loss = 0.0
correct_val = 0
total_val = 0

with torch.no_grad():
    for images, labels in validation_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_val += (predicted == labels).sum().item()
        total_val += labels.size(0)

val_loss = running_val_loss / total_val
val_accuracy = 100.0 * correct_val / total_val

elapsed = time.time() - start_time

# --- Print results ---
print(
    f"\nEpoch 1/1 | "
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_accuracy:.2f}% | "
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_accuracy:.2f}%"
)

# --- Save the model ---
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_index": train_dataset.class_to_index,
    },
    model_save_path,
)

print(f"\n========== Training Complete ==========\n")


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print(f"Training time:       {elapsed:.1f} seconds")
print(f"Training Accuracy:   {train_accuracy:.2f}%")
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Model saved to:      {model_save_path}")
print(f"Class mapping:       {train_dataset.class_to_index}")

# Confirm .pth file was created
if model_save_path.exists():
    size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] {model_save_path} exists ({size_mb:.1f} MB)")
else:
    print(f"\n[ERROR] {model_save_path} was NOT created!")
