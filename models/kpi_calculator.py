"""
KPI Calculator — OEE, MTBF, Availability, RUL
Computes industrial performance indicators from sensor history.

Author: Wassim BELAID
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


@dataclass
class KPIResult:
    machine_id: str
    oee: float               # Overall Equipment Effectiveness (%)
    availability: float      # Uptime / Total time (%)
    performance: float       # Actual speed / Ideal speed (%)
    quality: float           # Good parts / Total parts (%)
    mtbf: float              # Mean Time Between Failures (hours)
    mttr: float              # Mean Time To Repair (minutes)
    anomaly_rate: float      # % of anomalous readings
    rul_days: float          # Remaining Useful Life (days)
    health_score: float      # 0-100 overall health
    status: str              # "healthy", "degraded", "critical"


class KPICalculator:
    """
    Computes industrial KPIs from anomaly scores and sensor data.

    Usage:
        calc = KPICalculator("M01")
        calc.update(anomaly_score=0.1, is_anomaly=False)
        kpi = calc.compute()
    """

    def __init__(self, machine_id: str, target_oee: float = 85.0):
        self.machine_id = machine_id
        self.target_oee = target_oee
        self._scores: deque = deque(maxlen=500)
        self._anomalies: deque = deque(maxlen=500)
        self._downtime_events: List[float] = []       # durations in minutes
        self._repair_times: List[float] = []           # durations in minutes
        self._in_fault = False
        self._fault_start: Optional[float] = None
        self._total_samples = 0
        self._anomaly_samples = 0
        self._degradation_score = 0.0

    def update(self, anomaly_score: float, is_anomaly: bool, timestamp: float = None):
        """Feed one sample to the calculator."""
        self._scores.append(anomaly_score)
        self._anomalies.append(is_anomaly)
        self._total_samples += 1

        if is_anomaly:
            self._anomaly_samples += 1
            self._degradation_score = min(100.0, self._degradation_score + 0.5)
        else:
            self._degradation_score = max(0.0, self._degradation_score - 0.1)

        # Fault state machine
        import time
        now = timestamp or time.time()
        if is_anomaly and not self._in_fault:
            self._in_fault = True
            self._fault_start = now
        elif not is_anomaly and self._in_fault:
            if self._fault_start:
                duration_min = (now - self._fault_start) / 60
                self._downtime_events.append(duration_min)
                self._repair_times.append(min(duration_min, 30.0))
            self._in_fault = False
            self._fault_start = None

    def compute(self) -> KPIResult:
        """Compute all KPIs from current data."""
        if self._total_samples == 0:
            return self._default_kpi()

        # Anomaly rate
        anomaly_rate = self._anomaly_samples / self._total_samples * 100

        # Availability (based on fault events)
        n_faults = len(self._downtime_events)
        total_downtime = sum(self._downtime_events) / 60  # hours
        total_uptime_h = self._total_samples * 2 / 3600   # 2s per sample → hours
        total_time_h = total_uptime_h + total_downtime

        if total_time_h > 0:
            availability = (total_uptime_h / total_time_h) * 100
        else:
            availability = 100.0
        availability = round(min(100.0, max(0.0, availability)), 1)

        # MTBF
        if n_faults > 0 and total_uptime_h > 0:
            mtbf = round(total_uptime_h / n_faults, 2)
        else:
            mtbf = round(total_uptime_h, 2)

        # MTTR
        mttr = round(np.mean(self._repair_times), 1) if self._repair_times else 0.0

        # Performance (inverse of anomaly rate, with floor)
        performance = round(max(60.0, 100.0 - anomaly_rate * 3), 1)

        # Quality (based on recent score)
        recent_scores = list(self._scores)[-50:] if self._scores else [0]
        avg_score = np.mean(recent_scores)
        quality = round(max(70.0, 100.0 - avg_score * 80), 1)

        # OEE
        oee = round((availability / 100) * (performance / 100) * (quality / 100) * 100, 1)

        # RUL estimation (days)
        # Simple linear degradation model
        if self._degradation_score > 0:
            # At 100 degradation score → 0 days RUL
            rul_days = round(max(0.0, (100.0 - self._degradation_score) / 100 * 365), 1)
        else:
            rul_days = 365.0

        # Health score
        health_score = round(
            0.4 * availability +
            0.3 * performance +
            0.2 * quality +
            0.1 * min(100, rul_days / 3.65),
            1
        )

        # Status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "critical"

        return KPIResult(
            machine_id=self.machine_id,
            oee=oee,
            availability=availability,
            performance=performance,
            quality=quality,
            mtbf=mtbf,
            mttr=mttr,
            anomaly_rate=round(anomaly_rate, 2),
            rul_days=rul_days,
            health_score=health_score,
            status=status,
        )

    def _default_kpi(self) -> KPIResult:
        return KPIResult(
            machine_id=self.machine_id,
            oee=0.0, availability=0.0, performance=0.0,
            quality=0.0, mtbf=0.0, mttr=0.0,
            anomaly_rate=0.0, rul_days=365.0,
            health_score=100.0, status="healthy",
        )
