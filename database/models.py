"""
Database Models (SQLAlchemy ORM)
=================================
Defines the five tables used by the Smart Irrigation system:

    1. sensor_readings          – historical sensor data
    2. irrigation_predictions   – irrigation recommendation log
    3. crop_health_predictions  – crop health classification log
    4. alerts                   – system alerts / warnings
    5. system_config            – key-value runtime settings

All tables share the same declarative Base, so a single
``Base.metadata.create_all(engine)`` creates everything.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


# --------------------------------------------------
# Declarative base
# --------------------------------------------------

class Base(DeclarativeBase):
    """Shared base class for all ORM models."""
    pass


# --------------------------------------------------
# 1. Sensor Readings
# --------------------------------------------------

class SensorReading(Base):
    """
    Stores a single snapshot of sensor data (real or
    simulated).  The React dashboard will query this
    table for live readings and historical charts.
    """
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    soil_moisture = Column(Float, nullable=False)   # %
    temperature = Column(Float, nullable=False)      # °C
    humidity = Column(Float, nullable=False)          # %
    rainfall = Column(Float, nullable=False)          # mm/hr

    def __repr__(self):
        return (
            f"<SensorReading id={self.id} "
            f"moisture={self.soil_moisture} "
            f"temp={self.temperature}>"
        )


# --------------------------------------------------
# 2. Irrigation Predictions
# --------------------------------------------------

class IrrigationPrediction(Base):
    """
    Logs every irrigation recommendation produced by
    the recommendation engine.
    """
    __tablename__ = "irrigation_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    crop_type = Column(String(100), nullable=False)
    soil_type = Column(String(100), nullable=False)
    growth_stage = Column(String(100), nullable=False)
    ml_prediction = Column(Integer, nullable=False)       # 0 or 1
    confidence = Column(Float, nullable=True)              # may be null
    irrigation_required = Column(Boolean, nullable=False)
    reason = Column(Text, nullable=False)

    def __repr__(self):
        return (
            f"<IrrigationPrediction id={self.id} "
            f"irrigate={self.irrigation_required}>"
        )


# --------------------------------------------------
# 3. Crop Health Predictions
# --------------------------------------------------

class CropHealthPrediction(Base):
    """
    Logs every crop health classification result.
    """
    __tablename__ = "crop_health_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    image_path = Column(String(500), nullable=False)
    predicted_class = Column(String(200), nullable=False)
    confidence = Column(Float, nullable=False)

    def __repr__(self):
        return (
            f"<CropHealthPrediction id={self.id} "
            f"class={self.predicted_class}>"
        )


# --------------------------------------------------
# 4. Alerts
# --------------------------------------------------

class Alert(Base):
    """
    System alerts — e.g. low moisture warnings, sensor
    failures, irrigation schedule changes.
    """
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    alert_type = Column(String(100), nullable=False)     # e.g. "low_moisture"
    message = Column(Text, nullable=False)
    severity = Column(String(50), nullable=False)         # "info" / "warning" / "critical"
    acknowledged = Column(Boolean, nullable=False, default=False)

    def __repr__(self):
        return (
            f"<Alert id={self.id} type={self.alert_type} "
            f"severity={self.severity}>"
        )


# --------------------------------------------------
# 5. System Config
# --------------------------------------------------

class SystemConfig(Base):
    """
    Simple key-value store for runtime configuration
    (e.g. thresholds, schedule intervals).
    """
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(200), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)

    def __repr__(self):
        return f"<SystemConfig key={self.key}>"
