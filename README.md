# 📊 Industrial Monitoring Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Real-time industrial equipment monitoring with **AI-powered anomaly detection**, KPI tracking, predictive maintenance alerts and multi-sensor trend analysis.

Inspired by real maintenance work at **SIBEA (Béjaia, Algeria)** where early anomaly detection reduced machine downtime by **20%**.

---

## 🎯 Features

- **Real-time sensor dashboard** — temperature, vibration, pressure, current, speed
- **AI anomaly detection** — Isolation Forest + Z-Score statistical analysis
- **Predictive maintenance** — RUL (Remaining Useful Life) estimation
- **KPI tracking** — OEE, MTBF, availability, performance rate
- **Multi-machine support** — monitor 4 machines simultaneously
- **Alert system** — severity levels with maintenance recommendations
- **Historical trends** — 24h, 7d, 30d analysis

---

## 📁 Project Structure

```
industrial-monitoring-dashboard/
├── data/
│   ├── simulator.py        # Industrial sensor data simulator
│   └── loader.py           # Data loading and preprocessing
├── models/
│   ├── anomaly_detector.py # Isolation Forest + Z-Score detector
│   ├── rul_estimator.py    # Remaining Useful Life estimator
│   └── kpi_calculator.py   # OEE, MTBF, availability metrics
├── dashboard/
│   ├── app.py              # Main Streamlit application
│   └── components/
│       ├── gauges.py       # Real-time gauge components
│       ├── alerts.py       # Alert panel component
│       └── trends.py       # Trend chart component
├── utils/
│   └── config.py           # Machine configurations
├── tests/
│   └── test_models.py      # Unit tests
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/wassimbelaid05-EIE/industrial-monitoring-dashboard.git
cd industrial-monitoring-dashboard
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 🤖 AI Models

### Anomaly Detection — Isolation Forest
Unsupervised ML model trained on normal operating data. Detects multivariate anomalies across all sensors simultaneously with a configurable contamination rate.

### Statistical Control — Z-Score
Per-sensor statistical bounds (μ ± 3σ) with rolling window. Complementary to the ML model for interpretable alerts.

### Predictive Maintenance — RUL Estimation
Estimates Remaining Useful Life in days based on cumulative anomaly score and degradation trend.

---

## 📊 KPIs

| KPI | Target |
|-----|--------|
| OEE | > 85% |
| Availability | > 90% |
| Anomaly Rate | < 5% |
| RUL | > 30 days |

---

## 📈 Real Reference Results

Based on maintenance experience at SIBEA:
- ✅ **−20% machine downtime** through early anomaly detection
- ✅ **−15% maintenance incidents** via predictive alerts
- ✅ **+10% production performance** through optimized scheduling

---

## 👤 Author

**Wassim BELAID** — MSc Electrical Engineering, HES-SO Lausanne
[GitHub](https://github.com/wassimbelaid05-EIE)
