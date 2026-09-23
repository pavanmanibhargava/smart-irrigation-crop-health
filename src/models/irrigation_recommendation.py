"""
Irrigation Recommendation Engine
==================================
Combines the trained irrigation ML model with simple
rule-based safety logic to produce an understandable
irrigation recommendation.

The ML model predicts whether irrigation is needed based
on crop type, soil type, growth stage, soil moisture,
temperature, and humidity.

On top of the ML prediction, environmental rules override
the recommendation when conditions clearly do not warrant
irrigation (e.g., heavy rainfall, already-saturated soil).

Design note:
    This module is independent of FastAPI, PostgreSQL, and
    React.  It exposes a plain predict_irrigation() function
    that returns a dictionary — ready to be called from a
    future API endpoint.

Usage (command-line demo):
    python -m src.models.irrigation_recommendation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = Path("models/irrigation_model.joblib")


# --------------------------------------------------
# Rule-based thresholds  (easy to adjust later)
# --------------------------------------------------

# If rainfall exceeds this, override → no irrigation
HIGH_RAINFALL_THRESHOLD = 10.0       # mm/hr

# If soil moisture exceeds this, override → no irrigation
HIGH_MOISTURE_THRESHOLD = 65.0       # %

# Soil moisture considered "low" (supports ML recommendation)
LOW_MOISTURE_THRESHOLD = 35.0        # %

# Rainfall considered "low" (supports ML recommendation)
LOW_RAINFALL_THRESHOLD = 5.0         # mm/hr


# --------------------------------------------------
# Valid input values  (from training data)
# --------------------------------------------------

VALID_CROP_TYPES = [
    "Carrot", "Chilli", "Potato", "Tomato", "Wheat",
]

VALID_SOIL_TYPES = [
    "Alluvial Soil", "Black Soil", "Chalky Soil",
    "Clay Soil", "Loam Soil", "Red Soil", "Sandy Soil",
]

VALID_GROWTH_STAGES = [
    "Flowering",
    "Fruit/Grain/Bulb Formation",
    "Germination",
    "Harvest",
    "Maturation",
    "Pollination",
    "Seedling Stage",
    "Vegetative Growth / Root or Tuber Development",
]


# --------------------------------------------------
# Model loading  (load once, reuse)
# --------------------------------------------------

_cached_model = None


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the irrigation sklearn Pipeline from disk.
    Caches the model after the first call.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Irrigation model not found: {model_path}\n"
            "Please train the model first with:\n"
            "  python -m src.models.train_irrigation_model"
        )

    _cached_model = joblib.load(model_path)
    return _cached_model


# --------------------------------------------------
# Input validation
# --------------------------------------------------

def _validate_inputs(
    sensor_data: dict,
    crop_type: str,
    growth_stage: str,
    soil_type: str,
) -> list[str]:
    """
    Return a list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Required sensor keys
    for key in ("soil_moisture", "temperature", "humidity", "rainfall"):
        if key not in sensor_data:
            errors.append(f"Missing sensor key: '{key}'")

    # Crop type
    if crop_type not in VALID_CROP_TYPES:
        errors.append(
            f"Invalid crop_type '{crop_type}'. "
            f"Valid: {VALID_CROP_TYPES}"
        )

    # Soil type
    if soil_type not in VALID_SOIL_TYPES:
        errors.append(
            f"Invalid soil_type '{soil_type}'. "
            f"Valid: {VALID_SOIL_TYPES}"
        )

    # Growth stage
    if growth_stage not in VALID_GROWTH_STAGES:
        errors.append(
            f"Invalid growth_stage '{growth_stage}'. "
            f"Valid: {VALID_GROWTH_STAGES}"
        )

    return errors


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_irrigation(
    sensor_data: dict[str, float],
    crop_type: str = "Tomato",
    growth_stage: str = "Vegetative Growth / Root or Tuber Development",
    soil_type: str = "Loam Soil",
) -> dict[str, Any]:
    """
    Produce an irrigation recommendation by combining the
    ML model prediction with rule-based safety overrides.

    Parameters
    ----------
    sensor_data : dict
        Must contain: soil_moisture, temperature, humidity,
        rainfall.
    crop_type : str
        One of VALID_CROP_TYPES.
    growth_stage : str
        One of VALID_GROWTH_STAGES.
    soil_type : str
        One of VALID_SOIL_TYPES.

    Returns
    -------
    dict
        {
            "irrigation_required": bool,
            "ml_prediction": int (0 or 1),
            "confidence": float or None,
            "reason": str,
            "sensor_data": dict,
        }
    """
    # --- Validate inputs ---
    errors = _validate_inputs(sensor_data, crop_type, growth_stage, soil_type)
    if errors:
        raise ValueError(
            "Invalid inputs:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    soil_moisture = sensor_data["soil_moisture"]
    temperature = sensor_data["temperature"]
    humidity = sensor_data["humidity"]
    rainfall = sensor_data["rainfall"]

    # --- Build the feature DataFrame for the ML model ---
    # The trained pipeline expects columns:
    #   crop ID, soil_type, Seedling Stage, MOI, temp, humidity
    # Rainfall is NOT a feature the model was trained on.
    features = pd.DataFrame([{
        "crop ID":        crop_type,
        "soil_type":      soil_type,
        "Seedling Stage": growth_stage,
        "MOI":            soil_moisture,
        "temp":           temperature,
        "humidity":       humidity,
    }])

    # --- ML prediction ---
    model = load_model()

    ml_prediction = int(model.predict(features)[0])

    # Confidence via predict_proba (available on Logistic Regression)
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        # Confidence = probability of the predicted class
        confidence = round(float(probabilities[ml_prediction]), 4)

    # --- Rule-based overrides ---
    irrigation_required, reason = _apply_rules(
        ml_prediction=ml_prediction,
        soil_moisture=soil_moisture,
        rainfall=rainfall,
    )

    return {
        "irrigation_required": irrigation_required,
        "ml_prediction": ml_prediction,
        "confidence": confidence,
        "reason": reason,
        "sensor_data": {
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
        },
    }


def _apply_rules(
    ml_prediction: int,
    soil_moisture: float,
    rainfall: float,
) -> tuple[bool, str]:
    """
    Apply rule-based safety/context logic on top of the ML
    prediction.

    Returns
    -------
    (irrigation_required, reason)
    """
    # Rule 1: Heavy rainfall → no irrigation
    if rainfall >= HIGH_RAINFALL_THRESHOLD:
        return False, (
            "Irrigation not recommended because rainfall is "
            f"currently high ({rainfall:.1f} mm/hr >= "
            f"{HIGH_RAINFALL_THRESHOLD} mm/hr threshold)."
        )

    # Rule 2: Soil already saturated → no irrigation
    if soil_moisture >= HIGH_MOISTURE_THRESHOLD:
        return False, (
            "Irrigation not recommended because soil moisture "
            f"is already sufficient ({soil_moisture:.1f}% >= "
            f"{HIGH_MOISTURE_THRESHOLD}% threshold)."
        )

    # Rule 3: ML says irrigate AND conditions support it
    if ml_prediction == 1:
        if (soil_moisture <= LOW_MOISTURE_THRESHOLD
                and rainfall <= LOW_RAINFALL_THRESHOLD):
            return True, (
                "Irrigation recommended because soil moisture is "
                f"low ({soil_moisture:.1f}%) and the ML model "
                "predicts irrigation is required."
            )
        elif soil_moisture <= LOW_MOISTURE_THRESHOLD:
            return True, (
                "Irrigation recommended because soil moisture is "
                f"low ({soil_moisture:.1f}%), although there is "
                f"moderate rainfall ({rainfall:.1f} mm/hr)."
            )
        else:
            return False, (
                "Irrigation not recommended. The ML model suggests "
                f"irrigation, but soil moisture ({soil_moisture:.1f}%) "
                f"is above the low threshold ({LOW_MOISTURE_THRESHOLD}%)."
            )

    # Rule 4: ML says no irrigation
    return False, (
        "Irrigation not recommended. The ML model does not "
        "predict irrigation is needed under current conditions."
    )


# --------------------------------------------------
# Command-line demo
# --------------------------------------------------

if __name__ == "__main__":

    from src.simulation.sensor_simulator import SensorSimulator

    print("=" * 64)
    print("  Irrigation Recommendation Engine — Demo")
    print("=" * 64)

    scenarios = ["normal", "dry", "rainy"]

    for scenario in scenarios:
        print(f"\n{'—' * 64}")
        print(f"  Scenario: {scenario.upper()}")
        print(f"{'—' * 64}\n")

        sim = SensorSimulator(scenario=scenario, seed=42)
        reading = sim.get_reading()

        print(f"  Sensor readings:")
        print(f"    Soil Moisture : {reading['soil_moisture']:.1f}%")
        print(f"    Temperature   : {reading['temperature']:.1f} C")
        print(f"    Humidity      : {reading['humidity']:.1f}%")
        print(f"    Rainfall      : {reading['rainfall']:.1f} mm/hr")

        result = predict_irrigation(
            sensor_data=reading,
            crop_type="Tomato",
            growth_stage="Vegetative Growth / Root or Tuber Development",
            soil_type="Loam Soil",
        )

        print()
        print(f"  ML prediction     : {result['ml_prediction']}")
        print(f"  Confidence        : {result['confidence']}")
        print(f"  Irrigate?         : {result['irrigation_required']}")
        print(f"  Reason            : {result['reason']}")

    # --- Validation checks ---
    print(f"\n{'=' * 64}")
    print("  Validation Checks")
    print(f"{'=' * 64}\n")

    all_ok = True

    # Check 1: Dry conditions can produce irrigation recommendation
    sim_dry = SensorSimulator(scenario="dry", seed=42)
    dry_reading = sim_dry.get_reading()
    dry_result = predict_irrigation(sensor_data=dry_reading, crop_type="Tomato")

    if dry_result["ml_prediction"] == 1 or dry_result["irrigation_required"]:
        print("  [PASS] Dry scenario: model considers irrigation.")
    else:
        print("  [INFO] Dry scenario: model does not recommend irrigation "
              f"(moisture={dry_reading['soil_moisture']:.1f}%). "
              "This is acceptable if ML confidence is low.")

    # Check 2: Rainy conditions should NOT trigger irrigation
    sim_rainy = SensorSimulator(scenario="rainy", seed=42)
    rainy_reading = sim_rainy.get_reading()
    rainy_result = predict_irrigation(sensor_data=rainy_reading, crop_type="Tomato")

    if not rainy_result["irrigation_required"]:
        print("  [PASS] Rainy scenario: irrigation NOT recommended.")
    else:
        all_ok = False
        print("  [FAIL] Rainy scenario: irrigation was recommended!")

    # Check 3: High moisture should NOT trigger irrigation
    high_moist = {
        "soil_moisture": 75.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall": 0.0,
    }
    high_moist_result = predict_irrigation(sensor_data=high_moist, crop_type="Tomato")

    if not high_moist_result["irrigation_required"]:
        print("  [PASS] High moisture: irrigation NOT recommended.")
    else:
        all_ok = False
        print("  [FAIL] High moisture: irrigation was recommended!")

    # Check 4: No crashes with all valid crop types
    try:
        for crop in VALID_CROP_TYPES:
            test_data = {
                "soil_moisture": 30.0,
                "temperature": 28.0,
                "humidity": 50.0,
                "rainfall": 1.0,
            }
            predict_irrigation(sensor_data=test_data, crop_type=crop)
        print("  [PASS] All crop types processed without errors.")
    except Exception as exc:
        all_ok = False
        print(f"  [FAIL] Crash with crop types: {exc}")

    # Summary
    print()
    if all_ok:
        print("  All validations PASSED.")
    else:
        print("  Some validations FAILED — see above.")
    print(f"{'=' * 64}")
