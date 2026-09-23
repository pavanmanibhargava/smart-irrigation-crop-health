"""
Crop Health Inference Module
=============================
Loads the trained ResNet18 checkpoint and predicts the
disease/health class of a single tomato leaf image.

Returns a dictionary suitable for future FastAPI integration:

    {
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.93,
        "probabilities": {
            "Tomato___Bacterial_spot": 0.01,
            ...
        }
    }

Usage (command-line test):
    python -m src.models.predict_crop_health <image_path>
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# Default paths
# --------------------------------------------------

MODEL_PATH = Path("models/crop_health_resnet18.pth")


# --------------------------------------------------
# Preprocessing  (same as validation_transform)
# --------------------------------------------------

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(model_path=MODEL_PATH):
    """
    Load the trained ResNet18 crop-health model from a checkpoint.

    Parameters
    ----------
    model_path : str or Path
        Path to the .pth checkpoint file.

    Returns
    -------
    model : torch.nn.Module
        The ResNet18 model in eval mode, on CPU.
    class_to_index : dict
        Mapping of class name -> integer label.
    index_to_class : dict
        Reverse mapping of integer label -> class name.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Please train the model first with:\n"
            "  python -m src.models.train_crop_health_model"
        )

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    class_to_index = checkpoint["class_to_index"]
    num_classes = len(class_to_index)

    # Reverse mapping: index -> class name
    index_to_class = {v: k for k, v in class_to_index.items()}

    # Recreate the same ResNet18 architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode (disables dropout, batchnorm training)
    model.eval()

    return model, class_to_index, index_to_class


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_crop_health(image_path, model=None, index_to_class=None):
    """
    Predict the crop health class for a single leaf image.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image file.
    model : torch.nn.Module, optional
        A preloaded model.  If None, the default checkpoint is loaded.
    index_to_class : dict, optional
        Index-to-class-name mapping.  Required when model is provided.

    Returns
    -------
    dict
        {
            "predicted_class": str,
            "confidence": float,
            "probabilities": {class_name: float, ...}
        }
    """
    image_path = Path(image_path)

    # --- Validate input ---
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Load model if not provided ---
    if model is None:
        model, _, index_to_class = load_model()

    # --- Load and preprocess the image ---
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Could not open image: {image_path}\n"
            f"Error: {exc}"
        )

    input_tensor = inference_transform(image)

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    input_batch = input_tensor.unsqueeze(0)

    # --- Run inference ---
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs, dim=1)

    # --- Extract results ---
    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = index_to_class[predicted_index.item()]
    confidence_value = confidence.item()

    # Build full probability dictionary
    prob_dict = {
        index_to_class[i]: round(probabilities[0, i].item(), 4)
        for i in range(len(index_to_class))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence_value, 4),
        "probabilities": prob_dict,
    }


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m src.models.predict_crop_health <image_path>")
        print("\nExample:")
        print("  python -m src.models.predict_crop_health data/raw/crop_health/Tomato___healthy/image.JPG")
        sys.exit(1)

    test_image_path = sys.argv[1]

    print(f"Image: {test_image_path}")
    print(f"Model: {MODEL_PATH}")
    print()

    try:
        result = predict_crop_health(test_image_path)

        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence'] * 100:.2f}%")
        print()
        print("All probabilities:")
        for class_name, prob in sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "#" * int(prob * 40)
            print(f"  {class_name:<50s} {prob * 100:6.2f}%  {bar}")

    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
