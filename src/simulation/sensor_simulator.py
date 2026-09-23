"""
Sensor Simulator
=================
Simulates environmental and soil sensors for the smart
irrigation system.  Generates realistic, time-correlated
readings for soil moisture, temperature, humidity, and
rainfall across three weather scenarios.

Design note:
    This module is intentionally decoupled from FastAPI,
    database, and frontend layers.  When real hardware
    sensors are available they can expose the same
    get_reading() interface, so the rest of the system
    does not need to change.

Usage (command-line demo):
    python -m src.simulation.sensor_simulator
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any


# --------------------------------------------------
# Valid sensor ranges
# --------------------------------------------------

SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "soil_moisture": (10.0, 80.0),    # %
    "temperature":   (5.0, 45.0),     # °C
    "humidity":      (20.0, 100.0),   # %
    "rainfall":      (0.0, 50.0),     # mm/hr
}


# --------------------------------------------------
# Scenario profiles
# --------------------------------------------------
# Each profile defines (center, spread) for every sensor.
# - center: the value the random walk gravitates toward.
# - spread: standard deviation of per-step noise.

SCENARIO_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "normal": {
        "soil_moisture": (45.0, 5.0),
        "temperature":   (25.0, 3.0),
        "humidity":      (55.0, 5.0),
        "rainfall":      (2.0, 2.0),
    },
    "dry": {
        "soil_moisture": (18.0, 3.0),
        "temperature":   (38.0, 2.0),
        "humidity":      (25.0, 3.0),
        "rainfall":      (0.0, 0.5),
    },
    "rainy": {
        "soil_moisture": (68.0, 4.0),
        "temperature":   (18.0, 2.0),
        "humidity":      (90.0, 3.0),
        "rainfall":      (25.0, 8.0),
    },
}

VALID_SCENARIOS = list(SCENARIO_PROFILES.keys())

# How strongly the random walk is pulled back toward the
# scenario center each step.  0 = pure random walk,
# 1 = snap to center.  0.15 gives smooth, realistic drift.
MEAN_REVERSION_STRENGTH = 0.15

# Simulated interval between readings (seconds)
DEFAULT_INTERVAL_SECONDS = 60


# --------------------------------------------------
# SensorSimulator class
# --------------------------------------------------

class SensorSimulator:
    """
    Generates time-series sensor readings for a chosen
    weather scenario with reproducible, time-correlated
    values.

    Parameters
    ----------
    scenario : str
        One of "normal", "dry", or "rainy".
    seed : int
        Random seed for reproducibility.
    interval_seconds : int
        Simulated time gap between consecutive readings.

    Example
    -------
    >>> sim = SensorSimulator(scenario="dry", seed=42)
    >>> reading = sim.get_reading()
    >>> print(reading["soil_moisture"])
    """

    def __init__(
        self,
        scenario: str = "normal",
        seed: int = 42,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    ) -> None:

        # --- Validate scenario ---
        if scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from: {VALID_SCENARIOS}"
            )

        self.scenario = scenario
        self.seed = seed
        self.interval_seconds = interval_seconds

        self._rng = random.Random(seed)
        self._profile = SCENARIO_PROFILES[scenario]

        # Initialise the internal state at the scenario center
        self._state: dict[str, float] = {
            sensor: center
            for sensor, (center, _spread) in self._profile.items()
        }

        # Simulated clock starts at "now"
        self._timestamp = datetime.now(tz=timezone.utc)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def get_reading(self) -> dict[str, Any]:
        """
        Return a single sensor reading and advance the
        internal clock.

        Returns
        -------
        dict
            Keys: timestamp, soil_moisture, temperature,
            humidity, rainfall.
        """
        self._step()

        reading: dict[str, Any] = {
            "timestamp": self._timestamp.isoformat(),
        }

        for sensor in SENSOR_RANGES:
            reading[sensor] = round(self._state[sensor], 2)

        return reading

    def get_readings(self, count: int) -> list[dict[str, Any]]:
        """
        Return *count* consecutive sensor readings.

        Parameters
        ----------
        count : int
            Number of readings to generate (must be >= 1).

        Returns
        -------
        list[dict]
        """
        if count < 1:
            raise ValueError("count must be >= 1")

        return [self.get_reading() for _ in range(count)]

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _step(self) -> None:
        """
        Advance the simulator by one time step.

        The value for each sensor follows a mean-reverting
        random walk:

            new = old + reversion_toward_center + noise

        This produces smooth, realistic drift rather than
        purely random jumps.
        """
        # Advance simulated clock
        self._timestamp += timedelta(seconds=self.interval_seconds)

        for sensor, (center, spread) in self._profile.items():
            current = self._state[sensor]

            # Mean-reversion pull
            reversion = MEAN_REVERSION_STRENGTH * (center - current)

            # Gaussian noise
            noise = self._rng.gauss(0, spread)

            new_value = current + reversion + noise
            lo, hi = SENSOR_RANGES[sensor]
            new_value = max(lo, min(hi, new_value))

            self._state[sensor] = new_value


# --------------------------------------------------
# Validation helper
# --------------------------------------------------

def validate_reading(reading: dict[str, Any]) -> list[str]:
    """
    Check that a reading has the expected keys and that
    every sensor value is inside its valid range.

    Returns a list of error messages (empty = all OK).
    """
    errors: list[str] = []

    expected_keys = {"timestamp"} | set(SENSOR_RANGES.keys())
    missing = expected_keys - set(reading.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")

    for sensor, (lo, hi) in SENSOR_RANGES.items():
        value = reading.get(sensor)
        if value is None:
            continue
        if not (lo <= value <= hi):
            errors.append(
                f"{sensor}={value} out of range [{lo}, {hi}]"
            )

    return errors


# --------------------------------------------------
# Command-line demonstration
# --------------------------------------------------

if __name__ == "__main__":

    DEMO_COUNT = 5

    print("=" * 60)
    print("  Sensor Simulator — Demo")
    print("=" * 60)

    all_ok = True

    for scenario in VALID_SCENARIOS:
        print(f"\n--- Scenario: {scenario.upper()} (seed=42) ---\n")

        sim = SensorSimulator(scenario=scenario, seed=42)
        readings = sim.get_readings(DEMO_COUNT)

        for i, r in enumerate(readings, start=1):
            print(
                f"  [{i}] {r['timestamp']}  |  "
                f"Moisture: {r['soil_moisture']:5.1f}%  |  "
                f"Temp: {r['temperature']:5.1f}°C  |  "
                f"Humidity: {r['humidity']:5.1f}%  |  "
                f"Rain: {r['rainfall']:5.1f} mm/hr"
            )
            errors = validate_reading(r)
            if errors:
                all_ok = False
                for err in errors:
                    print(f"    [FAIL] {err}")

    # --- Reproducibility check ---
    print("\n--- Reproducibility check ---\n")

    sim_a = SensorSimulator(scenario="normal", seed=99)
    sim_b = SensorSimulator(scenario="normal", seed=99)

    readings_a = sim_a.get_readings(3)
    readings_b = sim_b.get_readings(3)

    if readings_a == readings_b:
        print("  [PASS] Same seed produces identical readings.")
    else:
        all_ok = False
        print("  [FAIL] Readings differ for the same seed!")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("  All validations PASSED.")
    else:
        print("  Some validations FAILED — see above.")
    print("=" * 60)
