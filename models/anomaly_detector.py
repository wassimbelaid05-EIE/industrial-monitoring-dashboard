"""
Anomaly Detection — Isolation Forest + Z-Score
Author: Wassim BELAID
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    method: str
    triggered_sensors: List[str]
    severity: str


class ZScoreDetector:
    def __init__(self, window: int = 50, threshold: float = 3.0):
        self.window = window
        self.threshold = threshold
        self._buffers: Dict[str, deque] = {}

    def update(self, sensor_name: str, value: float) -> Tuple[bool, float]:
        if sensor_name not in self._buffers:
            self._buffers[sensor_name] = deque(maxlen=self.window)
        buf = self._buffers[sensor_name]
        buf.append(value)
        if len(buf) < 10:
            return False, 0.0
        arr = np.array(buf)
        mu = np.mean(arr[:-1])
        sigma = np.std(arr[:-1])
        if sigma < 1e-6:
            return False, 0.0
        z = abs((value - mu) / sigma)
        return z > self.threshold, round(z, 3)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._is_trained = False
        self._training_buffer: List[List[float]] = []
        self._min_train_samples = 50

    def add_training_sample(self, features: List[float]):
        self._training_buffer.append(features)
        if len(self._training_buffer) >= self._min_train_samples and not self._is_trained:
            self._train()

    def _train(self):
        X = np.array(self._training_buffer)
        X_scaled = self._scaler.fit_transform(X)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_trained = True

    def predict(self, features: List[float]) -> Tuple[bool, float]:
        if not self._is_trained:
            return False, 0.0
        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        prediction = self._model.predict(X_scaled)[0]
        score = -self._model.score_samples(X_scaled)[0]
        score_normalized = float(np.clip(score + 0.5, 0, 1))
        return prediction == -1, round(score_normalized, 4)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_progress(self) -> float:
        return min(100.0, len(self._training_buffer) / self._min_train_samples * 100)


class AnomalyDetector:
    """
    Combined anomaly detector: Isolation Forest + Z-Score.

    Usage:
        detector = AnomalyDetector("M01")
        result = detector.analyze({"temperature": 95.0, "vibration": 8.5})
    """

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self._if_detector = IsolationForestDetector()
        self._zscore = ZScoreDetector()
        self._score_history: deque = deque(maxlen=200)
        self._anomaly_count = 0
        self._total_count = 0

    def analyze(self, sensor_readings: Dict[str, float]) -> AnomalyResult:
        self._total_count += 1
        features = list(sensor_readings.values())

        triggered = []
        z_scores = {}
        for name, value in sensor_readings.items():
            is_anom, z = self._zscore.update(name, value)
            z_scores[name] = z
            if is_anom:
                triggered.append(name)

        self._if_detector.add_training_sample(features)
        if_anomaly, if_score = self._if_detector.predict(features)

        zscore_anomaly = len(triggered) > 0
        is_anomaly = if_anomaly or zscore_anomaly

        zscore_max = max(z_scores.values()) / 6.0 if z_scores else 0
        combined_score = max(if_score, min(1.0, zscore_max))
        confidence = round(combined_score * 100, 1)

        if if_anomaly and zscore_anomaly:
            method = "combined"
        elif if_anomaly:
            method = "isolation_forest"
        elif zscore_anomaly:
            method = "zscore"
        else:
            method = "normal"

        if combined_score > 0.75:
            severity = "critical"
        elif combined_score > 0.45:
            severity = "warning"
        else:
            severity = "normal"

        if is_anomaly:
            self._anomaly_count += 1

        self._score_history.append(combined_score)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(combined_score, 4),
            confidence=confidence,
            method=method,
            triggered_sensors=triggered,
            severity=severity,
        )

    @property
    def anomaly_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return round(self._anomaly_count / self._total_count * 100, 2)

    @property
    def recent_score(self) -> float:
        if not self._score_history:
            return 0.0
        recent = list(self._score_history)[-20:]
        return round(float(np.mean(recent)), 4)

    @property
    def training_progress(self) -> float:
        return self._if_detector.training_progress

    def score_history(self) -> List[float]:
        return list(self._score_history)
