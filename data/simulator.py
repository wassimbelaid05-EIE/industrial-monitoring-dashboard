"""
Industrial Sensor Data Simulator
Generates realistic sensor data for industrial machines with anomaly injection.

Author: Wassim BELAID
"""

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta


@dataclass
class SensorConfig:
    """Configuration for one sensor."""
    name: str
    unit: str
    normal_mean: float
    normal_std: float
    min_val: float
    max_val: float
    warning_threshold: float
    critical_threshold: float
    noise_std: float = 0.1


@dataclass
class MachineConfig:
    """Configuration for one machine."""
    machine_id: str
    name: str
    machine_type: str
    sensors: Dict[str, SensorConfig]
    degradation_rate: float = 0.001  # per sample


# ── Machine definitions ───────────────────────────────────────────────────────

MACHINES = {
    "M01": MachineConfig(
        machine_id="M01",
        name="Compressor A",
        machine_type="Rotary Compressor",
        sensors={
            "temperature": SensorConfig("Temperature", "°C", 75, 5, 20, 150, 100, 130, 0.5),
            "vibration":   SensorConfig("Vibration", "mm/s", 2.5, 0.5, 0, 20, 7.1, 11.2, 0.1),
            "pressure":    SensorConfig("Pressure", "bar", 6.5, 0.3, 0, 10, 8.0, 9.0, 0.05),
            "current":     SensorConfig("Current", "A", 45, 3, 0, 100, 60, 75, 0.5),
        },
        degradation_rate=0.0008,
    ),
    "M02": MachineConfig(
        machine_id="M02",
        name="Pump B",
        machine_type="Centrifugal Pump",
        sensors={
            "temperature": SensorConfig("Temperature", "°C", 55, 4, 20, 120, 80, 100, 0.4),
            "vibration":   SensorConfig("Vibration", "mm/s", 1.8, 0.3, 0, 15, 4.5, 7.1, 0.08),
            "flow":        SensorConfig("Flow", "L/min", 120, 10, 0, 300, 80, 50, 1.0),
            "current":     SensorConfig("Current", "A", 32, 2, 0, 80, 45, 55, 0.3),
        },
        degradation_rate=0.0012,
    ),
    "M03": MachineConfig(
        machine_id="M03",
        name="Motor C",
        machine_type="Induction Motor",
        sensors={
            "temperature": SensorConfig("Temperature", "°C", 65, 5, 20, 130, 90, 110, 0.5),
            "vibration":   SensorConfig("Vibration", "mm/s", 1.5, 0.3, 0, 12, 4.5, 7.1, 0.07),
            "speed":       SensorConfig("Speed", "RPM", 1480, 20, 0, 2000, 1400, 1350, 2.0),
            "current":     SensorConfig("Current", "A", 28, 2, 0, 60, 38, 45, 0.3),
        },
        degradation_rate=0.0006,
    ),
    "M04": MachineConfig(
        machine_id="M04",
        name="Conveyor D",
        machine_type="Belt Conveyor",
        sensors={
            "temperature": SensorConfig("Temperature", "°C", 45, 3, 20, 100, 65, 80, 0.3),
            "vibration":   SensorConfig("Vibration", "mm/s", 3.0, 0.4, 0, 18, 6.3, 10.0, 0.1),
            "speed":       SensorConfig("Speed", "m/min", 25, 1, 0, 50, 18, 15, 0.2),
            "current":     SensorConfig("Current", "A", 18, 1.5, 0, 40, 25, 32, 0.2),
        },
        degradation_rate=0.001,
    ),
}


class IndustrialSimulator:
    """
    Simulates realistic industrial sensor data with:
    - Normal operating variations
    - Gradual degradation over time
    - Random anomaly injection
    - Sensor faults

    Usage:
        sim = IndustrialSimulator()
        data = sim.get_current_readings()   # latest values
        history = sim.get_history("M01", hours=1)
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._history: Dict[str, List[dict]] = {mid: [] for mid in MACHINES}
        self._degradation: Dict[str, float] = {mid: 0.0 for mid in MACHINES}
        self._anomaly_active: Dict[str, bool] = {mid: False for mid in MACHINES}
        self._anomaly_counter: Dict[str, int] = {mid: 0 for mid in MACHINES}
        self._tick = 0

        # Pre-generate 200 samples of history
        for _ in range(200):
            self._step()

    def _step(self):
        """Advance simulation by one tick."""
        self._tick += 1
        ts = datetime.now() - timedelta(seconds=(200 - self._tick) * 2)

        for mid, machine in MACHINES.items():
            # Update degradation
            self._degradation[mid] = min(1.0, self._degradation[mid] + machine.degradation_rate)
            deg = self._degradation[mid]

            # Random anomaly injection (5% chance per tick)
            if not self._anomaly_active[mid] and np.random.random() < 0.03:
                self._anomaly_active[mid] = True
                self._anomaly_counter[mid] = np.random.randint(3, 10)

            if self._anomaly_active[mid]:
                self._anomaly_counter[mid] -= 1
                if self._anomaly_counter[mid] <= 0:
                    self._anomaly_active[mid] = False
                anomaly = True
            else:
                anomaly = False

            # Generate sensor readings
            readings = {"timestamp": ts, "machine_id": mid, "anomaly_injected": anomaly}
            for skey, sensor in machine.sensors.items():
                base = sensor.normal_mean * (1 + deg * 0.3)  # drift with degradation
                noise = np.random.normal(0, sensor.noise_std)
                value = base + noise
                if anomaly:
                    value += np.random.uniform(sensor.normal_std * 3, sensor.normal_std * 6)
                value = np.clip(value, sensor.min_val, sensor.max_val)
                readings[skey] = round(value, 3)

            self._history[mid].append(readings)
            # Keep last 1000 samples
            if len(self._history[mid]) > 1000:
                self._history[mid].pop(0)

    def update(self):
        """Call this to advance simulation by one tick."""
        self._step()

    def get_current_readings(self) -> Dict[str, dict]:
        """Return latest readings for all machines."""
        self.update()
        result = {}
        for mid in MACHINES:
            if self._history[mid]:
                result[mid] = self._history[mid][-1].copy()
                result[mid]["machine_name"] = MACHINES[mid].name
                result[mid]["machine_type"] = MACHINES[mid].machine_type
                result[mid]["degradation_pct"] = round(self._degradation[mid] * 100, 1)
        return result

    def get_history(self, machine_id: str, samples: int = 100) -> pd.DataFrame:
        """Return historical data as DataFrame."""
        if machine_id not in self._history:
            return pd.DataFrame()
        data = self._history[machine_id][-samples:]
        return pd.DataFrame(data)

    def get_sensor_config(self, machine_id: str) -> Dict[str, SensorConfig]:
        return MACHINES[machine_id].sensors

    def get_all_history(self, samples: int = 100) -> Dict[str, pd.DataFrame]:
        return {mid: self.get_history(mid, samples) for mid in MACHINES}

    def inject_anomaly(self, machine_id: str, duration: int = 5):
        """Manually inject an anomaly for testing."""
        self._anomaly_active[machine_id] = True
        self._anomaly_counter[machine_id] = duration

    @property
    def machine_ids(self) -> List[str]:
        return list(MACHINES.keys())

    @property
    def machine_names(self) -> Dict[str, str]:
        return {mid: m.name for mid, m in MACHINES.items()}
