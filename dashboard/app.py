"""
Industrial Monitoring Dashboard
AI anomaly detection + predictive maintenance
Author: Wassim BELAID
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from data.simulator import IndustrialSimulator, MACHINES
from models.anomaly_detector import AnomalyDetector
from models.kpi_calculator import KPICalculator

st.set_page_config(page_title="Industrial Monitor", page_icon="📊", layout="wide")
st_autorefresh(interval=2000, key="monitor_refresh")

st.markdown("""
<style>
.metric-box{background:#1a1d2e;border-radius:10px;padding:16px;text-align:center;border:1px solid #2a2d3e;}
.alert-critical{background:#3d0000;border-left:4px solid #ff0000;padding:10px;border-radius:6px;margin:4px 0;}
.alert-warning{background:#3d2200;border-left:4px solid #ff8c00;padding:10px;border-radius:6px;margin:4px 0;}
.alert-normal{background:#003d1a;border-left:4px solid #00cc66;padding:10px;border-radius:6px;margin:4px 0;}
</style>""", unsafe_allow_html=True)

if "simulator" not in st.session_state:
    st.session_state.simulator = IndustrialSimulator()
    st.session_state.detectors = {mid: AnomalyDetector(mid) for mid in MACHINES}
    st.session_state.calculators = {mid: KPICalculator(mid) for mid in MACHINES}
    st.session_state.alert_log = []
    st.session_state.tick = 0

sim = st.session_state.simulator
detectors = st.session_state.detectors
calculators = st.session_state.calculators
st.session_state.tick += 1

readings = sim.get_current_readings()
results = {}
kpis = {}
for mid, data in readings.items():
    sensor_data = {k: v for k, v in data.items()
                   if k not in ("timestamp","machine_id","machine_name","machine_type","degradation_pct","anomaly_injected")}
    result = detectors[mid].analyze(sensor_data)
    calculators[mid].update(result.anomaly_score, result.is_anomaly)
    results[mid] = result
    kpis[mid] = calculators[mid].compute()
    if result.severity == "critical":
        st.session_state.alert_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "machine": data["machine_name"],
            "severity": result.severity,
            "sensors": ", ".join(result.triggered_sensors) or "ML model",
            "score": result.confidence,
        })
st.session_state.alert_log = st.session_state.alert_log[-50:]

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Industrial Monitor")
    st.caption("AI-powered predictive maintenance")
    st.divider()
    selected_machine = st.selectbox("🏭 Machine", list(MACHINES.keys()),
        format_func=lambda x: f"{x} — {MACHINES[x].name}")
    st.divider()
    if st.button("💥 Inject Anomaly", use_container_width=True, type="secondary"):
        sim.inject_anomaly(selected_machine, duration=8)
        st.toast(f"Anomaly injected!", icon="⚠️")
    st.divider()
    st.subheader("🤖 AI Model Training")
    for mid in MACHINES:
        prog = detectors[mid].training_progress
        st.caption(MACHINES[mid].name)
        st.progress(prog/100, text=f"{prog:.0f}%")
    st.divider()
    st.subheader("📋 Fleet Overview")
    for mid, kpi in kpis.items():
        icon = "🟢" if kpi.status == "healthy" else ("🟡" if kpi.status == "degraded" else "🔴")
        st.markdown(f"{icon} **{MACHINES[mid].name}** — OEE: {kpi.oee}%")

# HEADER
st.markdown("## 📊 Industrial Monitoring Dashboard")
st.caption(f"Real-time AI anomaly detection  |  Scan #{st.session_state.tick}  |  {datetime.now().strftime('%H:%M:%S')}")

# FLEET CARDS
cols = st.columns(4)
for i, (mid, kpi) in enumerate(kpis.items()):
    with cols[i]:
        result = results[mid]
        sc = {"healthy":"#00cc66","degraded":"#ff8c00","critical":"#ff3333"}[kpi.status]
        st.markdown(f"""<div class="metric-box">
            <h4 style="color:#aaa;margin:0">{MACHINES[mid].name}</h4>
            <h2 style="color:{sc};margin:4px 0">{kpi.health_score}%</h2>
            <p style="color:#888;margin:0;font-size:12px">Health Score</p>
            <hr style="border-color:#333;margin:8px 0">
            <span style="font-size:11px;color:#aaa">OEE: <b style="color:white">{kpi.oee}%</b></span><br>
            <span style="font-size:11px;color:#aaa">RUL: <b style="color:white">{kpi.rul_days}d</b></span><br>
            <span style="font-size:14px">{"🚨" if result.is_anomaly else "✅"}</span>
        </div>""", unsafe_allow_html=True)

st.divider()

# MACHINE DETAIL
mid = selected_machine
machine_data = readings[mid]
result = results[mid]
kpi = kpis[mid]
history = sim.get_history(mid, samples=100)
sensor_cfg = sim.get_sensor_config(mid)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"🏭 {MACHINES[mid].name} — {MACHINES[mid].machine_type}")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("OEE", f"{kpi.oee}%")
    k2.metric("Availability", f"{kpi.availability}%")
    k3.metric("MTBF", f"{kpi.mtbf}h")
    k4.metric("Anomaly Rate", f"{kpi.anomaly_rate}%")
    k5.metric("RUL", f"{kpi.rul_days}d")

    sensor_keys = [k for k in machine_data if k not in
                   ("timestamp","machine_id","machine_name","machine_type","degradation_pct","anomaly_injected")]
    s_cols = st.columns(len(sensor_keys))
    for i, skey in enumerate(sensor_keys):
        cfg = sensor_cfg[skey]
        val = machine_data[skey]
        s_cols[i].metric(f"{cfg.name} ({cfg.unit})", f"{val:.1f}", f"μ={cfg.normal_mean}", delta_color="off")

    if not history.empty:
        tab1, tab2 = st.tabs(["📈 Sensor Trends", "🤖 Anomaly Score"])
        with tab1:
            fig = make_subplots(rows=2, cols=2,
                subplot_titles=[sensor_cfg[k].name for k in sensor_keys[:4]],
                vertical_spacing=0.15)
            positions = [(1,1),(1,2),(2,1),(2,2)]
            colors = ["#cc0000","#2196F3","#FF9800","#9C27B0"]
            for i, skey in enumerate(sensor_keys[:4]):
                if skey not in history.columns: continue
                row, col = positions[i]
                cfg = sensor_cfg[skey]
                fig.add_trace(go.Scatter(y=history[skey].values, mode="lines",
                    name=cfg.name, line=dict(color=colors[i], width=1.5)), row=row, col=col)
                fig.add_hline(y=cfg.warning_threshold, line_dash="dash",
                    line_color="orange", row=row, col=col)
            fig.update_layout(template="plotly_dark", height=320,
                margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            score_hist = detectors[mid].score_history()
            if score_hist:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=score_hist, mode="lines",
                    line=dict(color="#cc0000", width=2), fill="tozeroy",
                    fillcolor="rgba(204,0,0,0.15)", name="Anomaly Score"))
                fig2.add_hline(y=0.45, line_dash="dash", line_color="orange",
                    annotation_text="Warning")
                fig2.add_hline(y=0.75, line_dash="dash", line_color="red",
                    annotation_text="Critical")
                fig2.update_layout(template="plotly_dark", height=280,
                    margin=dict(l=0,r=0,t=10,b=0), yaxis_range=[0,1])
                st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("🤖 AI Detection")
    sc = {"normal":"#00cc66","warning":"#ff8c00","critical":"#ff3333"}[result.severity]
    st.markdown(f"""<div style="background:#1a1d2e;border-radius:10px;padding:16px;border:2px solid {sc}">
        <h3 style="color:{sc};margin:0">{result.severity.upper()}</h3>
        <p style="color:#aaa;margin:4px 0">Score: <b style="color:white">{result.anomaly_score:.3f}</b></p>
        <p style="color:#aaa;margin:4px 0">Confidence: <b style="color:white">{result.confidence}%</b></p>
        <p style="color:#aaa;margin:4px 0">Method: <b style="color:white">{result.method}</b></p>
        <p style="color:#aaa;margin:4px 0">Anomaly Rate: <b style="color:white">{detectors[mid].anomaly_rate}%</b></p>
    </div>""", unsafe_allow_html=True)

    if result.triggered_sensors:
        st.warning(f"⚠️ {', '.join(result.triggered_sensors)}")

    st.subheader("💚 Health Gauge")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=kpi.health_score,
        gauge={"axis":{"range":[0,100]},"bar":{"color":sc},
               "steps":[{"range":[0,60],"color":"#3d0000"},
                        {"range":[60,80],"color":"#3d2200"},
                        {"range":[80,100],"color":"#003d1a"}],
               "threshold":{"line":{"color":"white","width":2},"value":85}},
        number={"suffix":"%","font":{"color":sc}},
    ))
    fig_g.update_layout(template="plotly_dark", height=220,
        margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_g, use_container_width=True)

    rul = kpi.rul_days
    rc = "#ff3333" if rul < 30 else ("#ff8c00" if rul < 90 else "#00cc66")
    st.markdown(f"""<div style="background:#1a1d2e;border-radius:10px;padding:12px;text-align:center">
        <p style="color:#aaa;margin:0;font-size:12px">REMAINING USEFUL LIFE</p>
        <h2 style="color:{rc};margin:4px 0">{rul} days</h2>
        <p style="color:#aaa;margin:0;font-size:11px">{"⚠️ MAINTENANCE URGENT" if rul < 30 else f"Next ~{int(rul)} days"}</p>
    </div>""", unsafe_allow_html=True)

    st.subheader("🚨 Alert Log")
    if st.session_state.alert_log:
        for alert in reversed(st.session_state.alert_log[-5:]):
            css = f"alert-{alert['severity']}"
            st.markdown(f'<div class="{css}"><b>{alert["time"]}</b> — {alert["machine"]}<br>'
                f'<small>{alert["sensors"]} | {alert["score"]:.1f}%</small></div>',
                unsafe_allow_html=True)
        if st.button("🗑️ Clear"):
            st.session_state.alert_log = []
    else:
        st.markdown('<div class="alert-normal">✅ No alerts</div>', unsafe_allow_html=True)
