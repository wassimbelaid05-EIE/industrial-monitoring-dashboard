"""
KPI Calculator — Industrial Monitoring Dashboard
Computes OEE, MTBF, MTTR, Availability and more.

Author: Wassim BELAID
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class KPISnapshot:
    machine_id: str
    timestamp: float
    availability: float       # %
    oee: float                # %
    mtbf_hours: float         # Mean Time Between Failures
    mttr_hours: float         # Mean Time To Repair
    anomaly_rate: float       # %
    total_readings: int
    total_anomalies: int
    fault_count: int
    uptime_hours: float

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "availability": round(self.availability, 1),
            "oee": round(self.oee, 1),
            "mtbf_h": round(self.mtbf_hours, 1),
            "mttr_h": round(self.mttr_hours, 2),
            "anomaly_rate": round(self.anomaly_rate, 2),
            "fault_count": self.fault_count,
            "uptime_h": round(self.uptime_hours, 2),
        }


class KPITracker:
    """
    Tracks KPIs for one machine over time.

    KPI Definitions (ISO 22400):
    - Availability  = Uptime / (Uptime + Downtime)
    - Performance   = Actual output / Theoretical max output (simulated 90-98%)
    - Quality       = Good parts / Total parts (simulated from anomaly rate)
    - OEE           = Availability × Performance × Quality
    - MTBF          = Total uptime / Number of failures
    - MTTR          = Total repair time / Number of repairs
    """

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self._start_time = time.time()
        self._total_readings = 0
        self._anomaly_readings = 0
        self._fault_count = 0
        self._downtime_seconds = 0.0
        self._repair_seconds = 0.0
        self._in_fault = False
        self._fault_start: Optional[float] = None
        self._performance = 0.95   # simulated
        self._last_fault_end: Optional[float] = None

    def update(self, is_anomaly: bool, is_fault: bool) -> None:
        """Update counters with latest reading."""
        self._total_readings += 1

        if is_anomaly:
            self._anomaly_readings += 1

        # Fault state machine
        if is_fault and not self._in_fault:
            self._in_fault = True
            self._fault_start = time.time()
            self._fault_count += 1

        elif not is_fault and self._in_fault:
            self._in_fault = False
            if self._fault_start:
                fault_duration = time.time() - self._fault_start
                self._downtime_seconds += fault_duration
                self._repair_seconds += fault_duration
                self._last_fault_end = time.time()
            self._fault_start = None

    def compute(self) -> KPISnapshot:
        """Compute current KPI snapshot."""
        elapsed = time.time() - self._start_time
        elapsed_hours = elapsed / 3600

        # Availability
        uptime_seconds = elapsed - self._downtime_seconds
        uptime_hours = uptime_seconds / 3600
        availability = (uptime_seconds / elapsed * 100) if elapsed > 0 else 100.0
        availability = max(0, min(100, availability))

        # MTBF
        if self._fault_count > 0:
            mtbf_hours = uptime_hours / self._fault_count
        else:
            mtbf_hours = uptime_hours  # no failures = all uptime

        # MTTR
        if self._fault_count > 0:
            mttr_hours = (self._repair_seconds / 3600) / self._fault_count
        else:
            mttr_hours = 0.0

        # Anomaly rate
        anomaly_rate = (self._anomaly_readings / self._total_readings * 100) if self._total_readings > 0 else 0.0

        # Quality (inverse of anomaly rate)
        quality = max(0, 100 - anomaly_rate * 2) / 100

        # OEE = A × P × Q
        oee = (availability / 100) * self._performance * quality * 100

        return KPISnapshot(
            machine_id=self.machine_id,
            timestamp=time.time(),
            availability=availability,
            oee=oee,
            mtbf_hours=mtbf_hours,
            mttr_hours=mttr_hours,
            anomaly_rate=anomaly_rate,
            total_readings=self._total_readings,
            total_anomalies=self._anomaly_readings,
            fault_count=self._fault_count,
            uptime_hours=uptime_hours,
        )


class PlantKPIManager:
    """Manages KPI trackers for all machines."""

    def __init__(self, machine_ids: List[str]):
        self._trackers = {mid: KPITracker(mid) for mid in machine_ids}

    def update(self, machine_id: str, is_anomaly: bool, is_fault: bool) -> None:
        if machine_id in self._trackers:
            self._trackers[machine_id].update(is_anomaly, is_fault)

    def get_kpi(self, machine_id: str) -> Optional[KPISnapshot]:
        if machine_id in self._trackers:
            return self._trackers[machine_id].compute()
        return None

    def get_all_kpis(self) -> Dict[str, KPISnapshot]:
        return {mid: t.compute() for mid, t in self._trackers.items()}

    def plant_oee(self) -> float:
        """Average OEE across all machines."""
        kpis = self.get_all_kpis()
        if not kpis:
            return 0.0
        return sum(k.oee for k in kpis.values()) / len(kpis)

    def plant_availability(self) -> float:
        kpis = self.get_all_kpis()
        if not kpis:
            return 0.0
        return sum(k.availability for k in kpis.values()) / len(kpis)
