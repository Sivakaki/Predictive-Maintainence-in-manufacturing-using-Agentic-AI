"""
Predictive Maintenance Dashboard
dashboard.py — Streamlit Web UI

Supports:
  - Real CSV data upload via sidebar
  - Simulated data as fallback

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from agents import OrchestratorAgent, SensorAgent


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PredictAI — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

.block-container { padding: 1.5rem 2rem; }

h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #58a6ff; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}

.severity-CRITICAL { color: #ff4444; font-weight: 700; }
.severity-HIGH     { color: #ff8c00; font-weight: 700; }
.severity-MEDIUM   { color: #ffd700; font-weight: 600; }
.severity-LOW      { color: #39d353; font-weight: 500; }
.severity-NORMAL   { color: #58a6ff; font-weight: 500; }

.stButton>button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(35,134,54,0.4); }

.stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem;
}

.task-row {
    background: #161b22;
    border-left: 4px solid #58a6ff;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.task-CRITICAL { border-left-color: #ff4444 !important; }
.task-HIGH     { border-left-color: #ff8c00 !important; }
.task-MEDIUM   { border-left-color: #ffd700 !important; }
.task-LOW      { border-left-color: #39d353 !important; }

.data-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
}
.data-badge-csv { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; }
.data-badge-sim { background: #f7816622; color: #f78166; border: 1px solid #f78166; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE — Data Source Management
# ─────────────────────────────────────────────

if "data_mode" not in st.session_state:
    st.session_state.data_mode = "SIMULATED"
    st.session_state.sensor_df = None
    st.session_state.fault_X = None
    st.session_state.fault_y = None
    st.session_state.orch_initialized = False


def initialize_orchestrator(sensor_df=None, fault_X=None, fault_y=None):
    """Create and bootstrap an OrchestratorAgent with the given data source."""
    orch = OrchestratorAgent()  # Always start in simulated mode at init

    if sensor_df is not None:
        # Load CSV data into the sensor agent
        orch.sensor_agent.load_readings_from_dataframe(sensor_df)

    orch.bootstrap()  # Trains analysis agent on historical (CSV or simulated)

    # If fault labels are uploaded, retrain diagnosis agent
    if fault_X is not None and fault_y is not None:
        orch.diagnosis_agent.train_from_data(fault_X, fault_y)

    # Run initial warm-up cycle
    orch.run_cycle(n_steps=5)
    return orch


@st.cache_resource(show_spinner="🔧 Bootstrapping AI agents...")
def load_orchestrator_simulated():
    """Load orchestrator in simulated mode (cached)."""
    return initialize_orchestrator()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PredictAI")
    st.markdown("*Agentic Predictive Maintenance*")
    st.divider()

    # ── DATA SOURCE SECTION ───────────────────
    st.markdown("### 📂 Data Source")

    sensor_file = st.file_uploader(
        "Upload Sensor CSV",
        type=["csv"],
        help="CSV with columns: machine_id, timestamp, temperature, vibration, pressure, rpm, current, oil_level",
        key="sensor_upload",
    )

    fault_file = st.file_uploader(
        "Upload Fault Labels CSV (optional)",
        type=["csv"],
        help="CSV with columns: temperature, vibration, pressure, rpm, current, oil_level, fault_type",
        key="fault_upload",
    )

    load_csv_btn = st.button("📥 Load CSV Data", use_container_width=True, disabled=(sensor_file is None))

    # Process CSV uploads
    if load_csv_btn and sensor_file is not None:
        try:
            from data_loader import load_sensor_from_uploaded_file, load_faults_from_uploaded_file

            sensor_df = load_sensor_from_uploaded_file(sensor_file)
            st.session_state.sensor_df = sensor_df
            st.session_state.data_mode = "CSV"

            if fault_file is not None:
                fault_X, fault_y = load_faults_from_uploaded_file(fault_file)
                st.session_state.fault_X = fault_X
                st.session_state.fault_y = fault_y

            # Re-initialize orchestrator with CSV data
            st.session_state.orch = initialize_orchestrator(
                sensor_df=st.session_state.sensor_df,
                fault_X=st.session_state.get("fault_X"),
                fault_y=st.session_state.get("fault_y"),
            )
            st.session_state.orch_initialized = True
            st.success(f"✅ Loaded {len(sensor_df):,} readings from {sensor_df['machine_id'].nunique()} machine(s)")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")

    # Data mode indicator
    if st.session_state.data_mode == "CSV":
        st.markdown('<span class="data-badge data-badge-csv">📊 CSV DATA</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="data-badge data-badge-sim">🎲 SIMULATED</span>', unsafe_allow_html=True)

    st.divider()

    # ── AGENT STATUS ──────────────────────────
    st.markdown("### 🤖 Agent Status")
    st.success("✅ SensorAgent — Online")
    st.success("✅ AnalysisAgent — Online")
    st.success("✅ DiagnosisAgent — Online")
    st.success("✅ MaintenanceAgent — Online")
    st.success("✅ InventoryAgent — Online")
    st.success("✅ OrchestratorAgent — Online")

    st.divider()


# ─────────────────────────────────────────────
# ORCHESTRATOR LOADING
# ─────────────────────────────────────────────
if st.session_state.orch_initialized and "orch" in st.session_state:
    orch = st.session_state.orch
else:
    orch = load_orchestrator_simulated()
    st.session_state.orch = orch
    st.session_state.orch_initialized = True


# ─────────────────────────────────────────────
# SIDEBAR — continued (needs orch)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Machines Monitored")
    for mid in orch.sensor_agent.MACHINES:
        st.markdown(f"🔵 `{mid}`")

    st.divider()
    selected_machine = st.selectbox(
        "Inspect Machine",
        options=orch.sensor_agent.MACHINES,
        index=0,
    )

    run_cycle_btn = st.button("▶ Run New Cycle", use_container_width=True)
    auto_refresh  = st.checkbox("Auto-refresh (10s)", value=False)

    st.divider()
    st.caption("Built with Python · scikit-learn · Streamlit · Plotly")


# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()

if run_cycle_btn:
    with st.spinner("Running monitoring cycle..."):
        orch.run_cycle(n_steps=2)
    st.success("Cycle complete!")
    st.rerun()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# ⚙️ Predictive Maintenance Dashboard")
mode_label = "📊 Real CSV Data" if st.session_state.data_mode == "CSV" else "🎲 Simulated Data"
st.markdown(f"*Last updated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` — Data Source: **{mode_label}***")
st.divider()


# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
summary = orch.get_system_summary()
all_tasks = orch.maintenance_agent.get_prioritized_queue()

total_readings  = len(orch.all_readings)
total_anomalies = sum(1 for a in orch.all_anomalies if a.is_anomalous)
critical_tasks  = sum(1 for t in all_tasks if t.priority == "CRITICAL")
avg_rul         = int(np.mean([v["estimated_rul"] for v in summary.values()])) if summary else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📡 Total Readings",   f"{total_readings:,}")
col2.metric("⚠️ Anomalies Detected", f"{total_anomalies}")
col3.metric("🔴 Critical Tasks",   f"{critical_tasks}")
col4.metric("📋 Tasks Queued",     f"{len(all_tasks)}")
col5.metric("⏳ Avg RUL (hrs)",    f"{avg_rul}")

st.divider()


# ─────────────────────────────────────────────
# MACHINE HEALTH OVERVIEW
# ─────────────────────────────────────────────
st.markdown("## 🏭 Fleet Health Overview")

SEVERITY_COLOR = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#39d353",
}

if summary:
    cols = st.columns(len(summary))
    for i, (mid, info) in enumerate(summary.items()):
        with cols[i]:
            color = SEVERITY_COLOR.get(info["severity"], "#58a6ff")
            health = info.get("health_score", 75)
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color}; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#8b949e;">{mid}</div>
                <div style="font-size:2rem; font-weight:700; color:{color}; margin:0.3rem 0;">{health:.0f}%</div>
                <div style="font-size:0.7rem; color:#8b949e;">Health Score</div>
                <hr style="border-color:#30363d; margin:0.5rem 0;">
                <div style="font-size:0.72rem; color:{color}; font-weight:600;">{info['severity']}</div>
                <div style="font-size:0.68rem; color:#8b949e; margin-top:0.2rem;">{info['latest_fault']}</div>
                <div style="font-size:0.68rem; color:#58a6ff; margin-top:0.2rem;">RUL: {info['estimated_rul']}h</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────
# SENSOR TIME SERIES — Selected Machine
# ─────────────────────────────────────────────
st.markdown(f"## 📈 Sensor Telemetry — `{selected_machine}`")

machine_readings = [r for r in orch.all_readings if r.machine_id == selected_machine]
machine_anomalies = [a for a in orch.all_anomalies if a.machine_id == selected_machine]

if machine_readings:
    df_r = pd.DataFrame([{
        "timestamp":   r.timestamp,
        "temperature": r.temperature,
        "vibration":   r.vibration,
        "pressure":    r.pressure,
        "rpm":         r.rpm,
        "current":     r.current,
        "oil_level":   r.oil_level,
    } for r in machine_readings]).sort_values("timestamp")

    df_a = pd.DataFrame([{
        "timestamp":     a.timestamp,
        "anomaly_score": a.anomaly_score,
        "is_anomalous":  a.is_anomalous,
    } for a in machine_anomalies]).sort_values("timestamp")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("🌡️ Temperature (°C)", "📳 Vibration (mm/s)",
                        "💨 Pressure (bar)",   "⚡ Current (A)",
                        "🔄 RPM",              "🛢️ Oil Level (%)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    PARAMS = [
        ("temperature", 1, 1, "#58a6ff", (60, 85)),
        ("vibration",   1, 2, "#f78166", (0.5, 3.0)),
        ("pressure",    2, 1, "#3fb950", (4.0, 7.0)),
        ("current",     2, 2, "#d2a8ff", (8.0, 12.0)),
        ("rpm",         3, 1, "#ffa657", (1400, 1600)),
        ("oil_level",   3, 2, "#79c0ff", (70, 100)),
    ]

    for param, row, col, color, (lo, hi) in PARAMS:
        fig.add_trace(go.Scatter(
            x=df_r["timestamp"], y=df_r[param],
            name=param, line=dict(color=color, width=1.8),
            fill="tozeroy", fillcolor=color.replace("ff", "22") if "#" in color else color,
            showlegend=False,
        ), row=row, col=col)

        # Danger zone
        fig.add_hline(y=hi * 1.15, line=dict(color="#ff4444", width=1, dash="dot"),
                      row=row, col=col)
        fig.add_hline(y=lo * 0.85, line=dict(color="#ff4444", width=1, dash="dot"),
                      row=row, col=col)

    fig.update_layout(
        height=650,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="JetBrains Mono", color="#e6edf3", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d")

    st.plotly_chart(fig, use_container_width=True)

    # Anomaly Score Timeline
    if not df_a.empty:
        st.markdown("### 🔍 Anomaly Score Timeline")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_a["timestamp"], y=df_a["anomaly_score"],
            fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
            line=dict(color="#ff4444", width=2),
            name="Anomaly Score",
        ))
        fig2.add_hline(y=0.5, line=dict(color="#ffd700", width=1.5, dash="dash"),
                       annotation_text="Alert Threshold")

        # Mark anomalous points
        anomalous = df_a[df_a["is_anomalous"]]
        if not anomalous.empty:
            fig2.add_trace(go.Scatter(
                x=anomalous["timestamp"], y=anomalous["anomaly_score"],
                mode="markers", marker=dict(color="#ff4444", size=8, symbol="circle"),
                name="Anomalous",
            ))

        fig2.update_layout(
            height=220,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(family="JetBrains Mono", color="#e6edf3", size=11),
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=True,
        )
        fig2.update_xaxes(gridcolor="#21262d")
        fig2.update_yaxes(gridcolor="#21262d", range=[0, 1.05])
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# INVENTORY STATUS
# ─────────────────────────────────────────────
st.markdown("## 📦 Inventory Status")
if hasattr(orch, "inventory_agent") and orch.inventory_agent:
    inv_status = orch.inventory_agent.get_inventory_status()
    if inv_status:
        df_inv = pd.DataFrame(inv_status)
        
        def highlight_stock(s):
            if s["Status"] == "Out of Stock" or s["Status"] == "On Order":
                return ['background-color: #4a0f0f'] * len(s)
            elif s["Status"] == "Low Stock":
                return ['background-color: #4a3f0f'] * len(s)
            else:
                return [''] * len(s)
                
        st.dataframe(df_inv.style.apply(highlight_stock, axis=1), use_container_width=True)
    else:
        st.info("Inventory system not tracking any parts.")
else:
    st.info("Inventory Agent is offline. Try checking 'Force Training Mode' or click Run New Cycle.")

st.divider()


# ─────────────────────────────────────────────
# MAINTENANCE TASK QUEUE
# ─────────────────────────────────────────────
st.markdown("## 📋 Maintenance Task Queue")

if all_tasks:
    for task in orch.maintenance_agent.get_prioritized_queue()[:20]:
        color = SEVERITY_COLOR.get(task.priority, "#58a6ff")
        st.markdown(f"""
        <div class="task-row task-{task.priority}">
            <span style="color:{color}; font-weight:700;">[{task.priority}]</span>
            &nbsp;
            <span style="color:#79c0ff;">{task.machine_id}</span>
            &nbsp;·&nbsp;
            <span style="color:#e6edf3;">{task.task_type}</span>
            &nbsp;·&nbsp;
            <span style="color:#8b949e;">Scheduled: {task.scheduled_for.strftime('%Y-%m-%d %H:%M')}</span>
            &nbsp;·&nbsp;
            <span style="color:#3fb950;">Downtime: {task.estimated_downtime_hours}h</span>
            <br>
            <span style="color:#6e7681; font-size:0.7rem;">{task.technician_notes}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No maintenance tasks scheduled. System is healthy!")

st.divider()


# ─────────────────────────────────────────────
# FAULT DISTRIBUTION PIE + SEVERITY BAR
# ─────────────────────────────────────────────
st.markdown("## 📊 Fault Analytics")

col_a, col_b = st.columns(2)

with col_a:
    all_diagnoses = orch.all_diagnoses
    if all_diagnoses:
        fault_counts = pd.Series([d.fault_type for d in all_diagnoses]).value_counts()
        fig3 = go.Figure(go.Pie(
            labels=fault_counts.index,
            values=fault_counts.values,
            hole=0.5,
            marker=dict(colors=px.colors.qualitative.Bold),
        ))
        fig3.update_layout(
            title="Fault Type Distribution",
            paper_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono", color="#e6edf3", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

with col_b:
    if all_tasks:
        sev_counts = pd.Series([t.priority for t in all_tasks]).value_counts()
        sev_colors = {"CRITICAL": "#ff4444", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#39d353"}
        fig4 = go.Figure(go.Bar(
            x=sev_counts.index,
            y=sev_counts.values,
            marker_color=[sev_colors.get(s, "#58a6ff") for s in sev_counts.index],
        ))
        fig4.update_layout(
            title="Tasks by Severity",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(family="JetBrains Mono", color="#e6edf3", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig4, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# DATA PREVIEW (CSV mode only)
# ─────────────────────────────────────────────
if st.session_state.data_mode == "CSV" and st.session_state.sensor_df is not None:
    with st.expander("📋 Preview Uploaded Sensor Data"):
        st.dataframe(st.session_state.sensor_df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(st.session_state.sensor_df):,} rows")

st.divider()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#6e7681; font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding:1rem 0;">
    ⚙️ PredictAI — Agentic Predictive Maintenance System &nbsp;|&nbsp;
    Built with Python · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO-LAUNCH HANDLER (for IDEs)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import subprocess
    # If the user tries to run this file directly via `python dashboard.py`, auto-redirect to streamlit.
    if "streamlit" not in sys.argv[0]:
        sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]]).returncode)


