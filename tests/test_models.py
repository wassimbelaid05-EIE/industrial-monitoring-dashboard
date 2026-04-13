"""
Unit Tests — Industrial Monitoring Dashboard
Author: Wassim BELAID
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from models.anomaly_detector import ZScoreDetector, IsolationForestDetector, AnomalyDetector
from models.kpi_calculator import KPICalculator
from data.simulator import IndustrialSimulator, MACHINES


# ── ZScore Tests ──────────────────────────────────────────────────────────────

class TestZScoreDetector:

    def test_no_anomaly_normal_values(self):
        det = ZScoreDetector(window=50, threshold=3.0)
        for _ in range(30):
            det.update("temp", np.random.normal(75, 2))
        is_anom, z = det.update("temp", 76.0)
        assert not is_anom

    def test_detects_spike(self):
        det = ZScoreDetector(window=50, threshold=3.0)
        for _ in range(30):
            det.update("temp", np.random.normal(75, 1))
        is_anom, z = det.update("temp", 200.0)  # extreme spike
        assert is_anom
        assert z > 3.0

    def test_insufficient_data_no_alarm(self):
        det = ZScoreDetector(window=50, threshold=3.0)
        det.update("temp", 1000.0)  # only 1 sample, no alarm
        is_anom, z = det.update("temp", 1000.0)
        assert not is_anom  # < 10 samples required

    def test_multiple_sensors_independent(self):
        det = ZScoreDetector()
        for _ in range(60):
            det.update("temp", 75.0)
            det.update("vib", 2.5)
        is_anom_t, _ = det.update("temp", 76.0)
        for _ in range(5):
            is_anom_v, _ = det.update("vib", 500.0)
        assert not is_anom_t
        assert is_anom_v


# ── Isolation Forest Tests ────────────────────────────────────────────────────

class TestIsolationForestDetector:

    def test_not_trained_initially(self):
        det = IsolationForestDetector()
        assert not det.is_trained

    def test_trains_after_min_samples(self):
        det = IsolationForestDetector()
        for _ in range(50):
            det.add_training_sample([75.0, 2.5, 6.5, 45.0])
        assert det.is_trained

    def test_training_progress(self):
        det = IsolationForestDetector()
        assert det.training_progress == 0.0
        for _ in range(25):
            det.add_training_sample([75.0, 2.5])
        assert det.training_progress == 50.0

    def test_predicts_after_training(self):
        det = IsolationForestDetector()
        for _ in range(60):
            det.add_training_sample([75.0 + np.random.normal(0, 1), 2.5 + np.random.normal(0, 0.1)])
        is_anom, score = det.predict([75.0, 2.5])
        assert isinstance(is_anom, (bool, __import__('numpy').bool_))
        assert 0.0 <= score <= 1.0

    def test_returns_false_when_not_trained(self):
        det = IsolationForestDetector()
        is_anom, score = det.predict([75.0, 2.5])
        assert not is_anom
        assert score == 0.0


# ── AnomalyDetector Tests ─────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_analyze_returns_result(self):
        det = AnomalyDetector("M01")
        result = det.analyze({"temp": 75.0, "vib": 2.5, "pressure": 6.5, "current": 45.0})
        assert result is not None
        assert result.severity in ("normal", "warning", "critical")
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_anomaly_rate_starts_zero(self):
        det = AnomalyDetector("M01")
        assert det.anomaly_rate == 0.0

    def test_anomaly_rate_increases(self):
        det = AnomalyDetector("M01")
        for _ in range(100):
            det.analyze({"temp": 75.0, "vib": 2.5})
        for _ in range(10):
            det.analyze({"temp": 500.0, "vib": 100.0})  # extreme anomaly
        assert det.anomaly_rate > 0.0

    def test_severity_levels(self):
        det = AnomalyDetector("M01")
        result = det.analyze({"temp": 75.0, "vib": 2.5})
        assert result.severity in ("normal", "warning", "critical")

    def test_score_history_fills(self):
        det = AnomalyDetector("M01")
        for _ in range(10):
            det.analyze({"temp": 75.0, "vib": 2.5})
        assert len(det.score_history()) == 10

    def test_training_progress(self):
        det = AnomalyDetector("M01")
        assert 0.0 <= det.training_progress <= 100.0


# ── KPICalculator Tests ───────────────────────────────────────────────────────

class TestKPICalculator:

    def test_compute_with_no_data(self):
        calc = KPICalculator("M01")
        kpi = calc.compute()
        assert kpi.machine_id == "M01"
        assert kpi.health_score == 100.0

    def test_oee_in_range(self):
        calc = KPICalculator("M01")
        for _ in range(50):
            calc.update(0.1, False)
        kpi = calc.compute()
        assert 0.0 <= kpi.oee <= 100.0

    def test_availability_in_range(self):
        calc = KPICalculator("M01")
        for _ in range(50):
            calc.update(0.2, True)
        kpi = calc.compute()
        assert 0.0 <= kpi.availability <= 100.0

    def test_status_healthy_with_normal_data(self):
        calc = KPICalculator("M01")
        for _ in range(100):
            calc.update(0.05, False)
        kpi = calc.compute()
        assert kpi.status in ("healthy", "degraded", "critical")

    def test_rul_decreases_with_anomalies(self):
        calc1 = KPICalculator("M01")
        calc2 = KPICalculator("M01")
        for _ in range(100):
            calc1.update(0.05, False)
            calc2.update(0.9, True)
        kpi1 = calc1.compute()
        kpi2 = calc2.compute()
        assert kpi1.rul_days > kpi2.rul_days

    def test_anomaly_rate_computed(self):
        calc = KPICalculator("M01")
        for _ in range(90):
            calc.update(0.05, False)
        for _ in range(10):
            calc.update(0.9, True)
        kpi = calc.compute()
        assert abs(kpi.anomaly_rate - 10.0) < 1.0


# ── Simulator Tests ───────────────────────────────────────────────────────────

class TestIndustrialSimulator:

    def test_initializes_with_history(self):
        sim = IndustrialSimulator()
        for mid in MACHINES:
            df = sim.get_history(mid)
            assert len(df) > 0

    def test_get_current_readings_all_machines(self):
        sim = IndustrialSimulator()
        readings = sim.get_current_readings()
        assert set(readings.keys()) == set(MACHINES.keys())

    def test_readings_have_sensor_values(self):
        sim = IndustrialSimulator()
        readings = sim.get_current_readings()
        for mid, data in readings.items():
            assert "machine_name" in data
            assert "degradation_pct" in data

    def test_inject_anomaly(self):
        sim = IndustrialSimulator()
        sim.inject_anomaly("M01", duration=3)
        # Just verify it doesn't crash
        readings = sim.get_current_readings()
        assert "M01" in readings

    def test_history_grows_with_updates(self):
        sim = IndustrialSimulator()
        before = len(sim.get_history("M01"))
        sim.get_current_readings()  # triggers update
        after = len(sim.get_history("M01"))
        assert after >= before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
