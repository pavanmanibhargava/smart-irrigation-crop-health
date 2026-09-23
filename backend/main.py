"""
FastAPI Backend for Smart Irrigation System
===========================================
Provides REST APIs for:
- Sensor Simulation
- Irrigation Recommendation
- Crop Health Prediction

Usage:
    python -m uvicorn backend.main:app --reload
"""

import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing modules
from src.simulation.sensor_simulator import SensorSimulator
from src.models.irrigation_recommendation import predict_irrigation
from src.models.predict_crop_health import predict_crop_health

# Initialize FastAPI App
app = FastAPI(
    title="Smart Irrigation API",
    description="Backend API for Smart Irrigation and Crop Health Capstone",
    version="1.0.0"
)

# Enable CORS for local React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# State & Globals
# --------------------------------------------------

# Instantiate a single sensor simulator (normal scenario by default)
# Using a fixed seed for reproducible gradual drift across API calls
simulator = SensorSimulator(scenario="normal", seed=42)

# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "message": "Smart Irrigation API is running"
    }


@app.get("/api/sensors/current")
def get_current_sensors():
    """Get a current reading from the simulated sensors."""
    try:
        reading = simulator.get_reading()
        return reading
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/irrigation/recommendation")
def get_irrigation_recommendation():
    """Get an irrigation recommendation based on current sensor readings."""
    try:
        # 1. Get current sensor data
        sensor_data = simulator.get_reading()
        
        # 2. Define default crop configuration for demo
        crop_type = "Tomato"
        soil_type = "Loam Soil"
        growth_stage = "Vegetative Growth / Root or Tuber Development"
        
        # 3. Get prediction
        recommendation = predict_irrigation(
            sensor_data=sensor_data,
            crop_type=crop_type,
            growth_stage=growth_stage,
            soil_type=soil_type
        )
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crop-health/predict")
async def predict_crop_health_endpoint(file: UploadFile = File(...)):
    """
    Upload an image of a tomato leaf to get a disease prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create temp directory if it doesn't exist
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file temporarily
    temp_path = temp_dir / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Run prediction using existing module
        result = predict_crop_health(str(temp_path))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass


@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    """
    Returns a combined payload for the dashboard overview.
    Contains current sensors, irrigation recommendation, and system status.
    """
    try:
        # Get recommendation (which includes sensor data internally)
        irrigation_rec = get_irrigation_recommendation()
        
        summary = {
            "system_status": {
                "state": "ONLINE",
                "mode": "AUTO"
            },
            "current_sensors": irrigation_rec["sensor_data"],
            "irrigation_recommendation": {
                "irrigation_required": irrigation_rec["irrigation_required"],
                "ml_prediction": irrigation_rec["ml_prediction"],
                "confidence": irrigation_rec["confidence"],
                "reason": irrigation_rec["reason"]
            },
            "models_status": {
                "irrigation_model": "LOADED",
                "crop_health_model": "LOADED"
            }
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
