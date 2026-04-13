"""
Unit Tests — Industrial Monitoring Dashboard
Author: Wassim BELAID
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from data.simulator import PlantSimulator, MachineSimulator, MACHINES, MachineState
from models.anomaly_detector import AnomalyDetector, MultiMachineDetector
from utils.kpi import KPITracker, PlantKPIManager


# ── Simulator Tests ───────────────────────────────────────────────────────────

class TestMachineSimulator:

    def test_reading_has_all_fields(self):
        sim = MachineSimulator("M01")
        r = sim.read()
        assert r.machine_id == "M01"
        assert r.vibration >= 0
        assert r.temperature >= 0
        assert r.pressure >= 0
        assert r.current >= 0

    def test_readings_are_positive(self):
        sim = MachineSimulator("M01")
        for _ in range(20):
            r = sim.read()
            assert r.vibration >= 0
            assert r.temperature >= 0
            assert r.current >= 0

    def test_history_grows(self):
        sim = MachineSimulator("M01")
        for _ in range(10):
            sim.read()
        assert len(sim.get_history()) == 10

    def test_history_capped_at_500(self):
        sim = MachineSimulator("M01")
        for _ in range(600):
            sim.read()
        assert len(sim.get_history()) <= 500

    def test_rul_decreases_with_age(self):
        sim = MachineSimulator("M01")
        sim.age_hours = 0
        rul1 = sim.rul_estimate()
        sim.age_hours = 1000
        rul2 = sim.rul_estimate()
        assert rul2 < rul1

    def test_to_dict_has_correct_keys(self):
        sim = MachineSimulator("M01")
        r = sim.read()
        d = r.to_dict()
        assert "vibration" in d
        assert "temperature" in d
        assert "pressure" in d
        assert "current" in d
        assert "state" in d


class TestPlantSimulator:

    def test_reads_all_machines(self):
        plant = PlantSimulator()
        readings = plant.read_all()
        assert set(readings.keys()) == set(MACHINES.keys())

    def test_training_data_generation(self):
        plant = PlantSimulator()
        data = plant.generate_training_data(n_samples=10)
        assert len(data) == 40  # 10 samples × 4 machines
        assert "vibration" in data[0]
        assert "temperature" in data[0]


# ── Anomaly Detector Tests ────────────────────────────────────────────────────

class TestAnomalyDetector:

    def setup_method(self):
        self.detector = AnomalyDetector(contamination=0.05)
        self.normal_data = [
            {"vibration": 2.5 + np.random.normal(0, 0.1),
             "temperature": 65.0 + np.random.normal(0, 1.0),
             "pressure": 4.5 + np.random.normal(0, 0.1),
             "current": 15.0 + np.random.normal(0, 0.3)}
            for _ in range(200)
        ]

    def test_train_sets_trained_flag(self):
        self.detector.train(self.normal_data)
        assert self.detector.is_trained

    def test_predict_returns_score_and_bool(self):
        self.detector.train(self.normal_data)
        reading = {"vibration": 2.5, "temperature": 65.0, "pressure": 4.5, "current": 15.0}
        score, is_anomaly = self.detector.predict(reading)
        assert 0.0 <= score <= 1.0
        assert isinstance(is_anomaly, bool)

    def test_normal_reading_low_score(self):
        self.detector.train(self.normal_data)
        reading = {"vibration": 2.5, "temperature": 65.0, "pressure": 4.5, "current": 15.0}
        score, _ = self.detector.predict(reading)
        assert score < 0.7

    def test_fault_reading_high_score(self):
        self.detector.train(self.normal_data)
        # Extreme fault values
        fault = {"vibration": 15.0, "temperature": 150.0, "pressure": 0.1, "current": 50.0}
        score, is_anomaly = self.detector.predict(fault)
        assert score > 0.5 or is_anomaly

    def test_predict_without_training_raises(self):
        with pytest.raises(RuntimeError):
            self.detector.predict({"vibration": 2.5, "temperature": 65.0, "pressure": 4.5, "current": 15.0})

    def test_score_between_0_and_1(self):
        self.detector.train(self.normal_data)
        for _ in range(20):
            reading = {
                "vibration": abs(np.random.normal(2.5, 2.0)),
                "temperature": abs(np.random.normal(65, 20)),
                "pressure": abs(np.random.normal(4.5, 1.0)),
                "current": abs(np.random.normal(15, 5)),
            }
            score, _ = self.detector.predict(reading)
            assert 0.0 <= score <= 1.0


class TestMultiMachineDetector:

    def test_train_all_machines(self):
        plant = PlantSimulator()
        raw = plant.generate_training_data(50)
        mmd = MultiMachineDetector()
        training = {mid: raw[i::4] for i, mid in enumerate(MACHINES.keys())}
        mmd.train_all(training)
        for mid in MACHINES:
            assert mmd.is_trained(mid)

    def test_predict_untrained_returns_zero(self):
        mmd = MultiMachineDetector()
        score, is_anomaly = mmd.predict("M01", {"vibration": 2.5, "temperature": 65.0, "pressure": 4.5, "current": 15.0})
        assert score == 0.0
        assert is_anomaly is False


# ── KPI Tests ─────────────────────────────────────────────────────────────────

class TestKPITracker:

    def test_initial_availability_100(self):
        import time
        tracker = KPITracker("M01")
        time.sleep(0.01)
        tracker.update(False, False)
        kpi = tracker.compute()
        assert kpi.availability > 95.0

    def test_fault_increments_counter(self):
        tracker = KPITracker("M01")
        tracker.update(False, True)
        tracker.update(False, False)
        kpi = tracker.compute()
        assert kpi.fault_count == 1

    def test_anomaly_rate_increases(self):
        tracker = KPITracker("M01")
        for _ in range(10):
            tracker.update(False, False)
        for _ in range(5):
            tracker.update(True, False)
        kpi = tracker.compute()
        assert kpi.anomaly_rate > 0

    def test_oee_between_0_and_100(self):
        tracker = KPITracker("M01")
        for _ in range(20):
            tracker.update(False, False)
        kpi = tracker.compute()
        assert 0 <= kpi.oee <= 100

    def test_kpi_to_dict(self):
        tracker = KPITracker("M01")
        tracker.update(False, False)
        d = tracker.compute().to_dict()
        assert "availability" in d
        assert "oee" in d
        assert "fault_count" in d


class TestPlantKPIManager:

    def test_manages_all_machines(self):
        mgr = PlantKPIManager(list(MACHINES.keys()))
        for mid in MACHINES:
            mgr.update(mid, False, False)
        kpis = mgr.get_all_kpis()
        assert set(kpis.keys()) == set(MACHINES.keys())

    def test_plant_oee_average(self):
        mgr = PlantKPIManager(list(MACHINES.keys()))
        for mid in MACHINES:
            for _ in range(5):
                mgr.update(mid, False, False)
        oee = mgr.plant_oee()
        assert 0 <= oee <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
