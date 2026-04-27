# Predictive Maintenance: Agentic AI for Industrial Machine Health
### Industrial IoT | Multi-Agent Pipeline | Python

## Overview

This project implements a full **multi-agent AI pipeline** for industrial machine health monitoring, combining **anomaly detection** and **fault classification** to enable proactive maintenance scheduling. The system ingests real or simulated sensor data and coordinates specialized agents — from raw signal collection through to maintenance task creation and spare-parts reservation.

The entire workflow is modular and self-contained: from synthetic sensor data generation to statistical anomaly detection, ML-based fault diagnosis, and automated maintenance scheduling with live inventory tracking.

---

## 🔬 Methodology

| Stage | Technique | Output |
|---|---|---|
| Data Preparation | C-MAPSS dataset scaling & mapping | `cmaps_sensor_data.csv`, `cmaps_fault_labels.csv` |
| Anomaly Detection | Isolation Forest (n_estimators=200) | `AnomalyReport` per reading |
| Fault Classification | Random Forest (n_estimators=150) | `DiagnosisReport` per reading |
| Maintenance Scheduling | Priority-based task queue | `MaintenanceTask` objects |
| Inventory Management | Stock check & reservation logic | Live inventory state |
| Visualization | Streamlit multi-page dashboard | Interactive web UI |

---

## 🤖 Agent Architecture

```
SensorAgent → AnalysisAgent → DiagnosisAgent → MaintenanceAgent
                                                       ↕
                                               InventoryAgent
                    ↑
            OrchestratorAgent (coordinates all)
```

| Agent | File | Role |
|---|---|---|
| **SensorAgent** | `sensor_agent.py` | Reads sensor data from CSV or simulation |
| **AnalysisAgent** | `analysis_agent.py` | Detects anomalies via Isolation Forest |
| **DiagnosisAgent** | `diagnosis_agent.py` | Classifies fault type via Random Forest |
| **MaintenanceAgent** | `maintenance_agent.py` | Schedules repair tasks by priority |
| **InventoryAgent** | `inventory_agent.py` | Checks and reserves spare parts |
| **OrchestratorAgent** | `orchestrator_agent.py` | Coordinates the full pipeline |

---

## 📊 Fault Types Classified (7 total)

1. **Normal Operation** – No intervention required; routine monitoring continues
2. **Bearing Wear** – High vibration; bearing inspection and replacement scheduled
3. **Overheating** – Elevated temperature; cooling system check triggered
4. **Lubrication Failure** – Low oil level; immediate top-up and filter check
5. **Imbalance / Misalignment** – Shaft alignment and balance check dispatched
6. **Electrical Fault** – High current draw; certified technician diagnostic
7. **Pressure Drop** – Below-threshold pressure; seals and valves inspected

---

## 🛠️ Tech Stack

- **Python 3.x**
- `numpy`, `pandas` — data generation & manipulation
- `scikit-learn` — IsolationForest, RandomForestClassifier, StandardScaler, Pipeline
- `joblib` — model persistence (`.pkl` files)
- `streamlit` — interactive monitoring dashboard
- `matplotlib`, `seaborn` — supporting visualizations

---

## 🚀 Usage

```bash
pip install numpy pandas scikit-learn joblib streamlit matplotlib seaborn
```

**Step 1 — (Optional) Prepare real C-MAPSS data:**
```bash
python prep_cmapss.py
```

**Step 2 — Train models:**
```bash
python train_models.py
```

**Step 3 — Run the pipeline:**
```bash
# Simulated data
python orchestrator_agent.py

# Real CSV data
python orchestrator_agent.py --sensor-csv cmaps_sensor_data.csv --labels-csv cmaps_fault_labels.csv
```

**Step 4 — Launch the dashboard:**
```bash
streamlit run dashboard.py
```

Trained models are saved as `model_analysis.pkl` and `model_diagnosis.pkl`. On subsequent runs they are loaded from disk automatically, skipping retraining.

---

## 📁 Project Files

```
├── models.py               # Shared dataclasses (SensorReading, AnomalyReport, etc.)
├── data_loader.py          # CSV ingestion and schema validation
├── sensor_agent.py         # Sensor data collection (CSV or simulated)
├── analysis_agent.py       # Anomaly detection — Isolation Forest
├── diagnosis_agent.py      # Fault classification — Random Forest
├── maintenance_agent.py    # Priority-based maintenance scheduling
├── inventory_agent.py      # Spare parts tracking and reservation
├── orchestrator_agent.py   # Pipeline coordinator + CLI entry point
├── agents.py               # Convenience re-exports
├── train_models.py         # Standalone model training script
├── prep_cmapss.py          # NASA C-MAPSS dataset preprocessor
├── dashboard.py            # Streamlit monitoring dashboard
└── test_all.py             # Smoke tests for all agents
```
