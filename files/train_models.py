"""
Predictive Maintenance — Dedicated Training Script
train_models.py

Explicitly loads CSV data, trains the agent ML models, 
and persists them to disk as .pkl files.
"""

import os
from sensor_agent import SensorAgent
from analysis_agent import AnalysisAgent
from diagnosis_agent import DiagnosisAgent

SENSOR_CSV = "cmaps_sensor_data.csv"
LABELS_CSV = "cmaps_fault_labels.csv"

def train():
    print("=========================================")
    print("  Predictive Maintenance — Model Trainer")
    print("=========================================\n")

    # 1. Initialize Sensor Agent to get historical data
    print("[1/3] Loading normal operation baseline data...")
    if not os.path.exists(SENSOR_CSV):
        raise FileNotFoundError(f"Missing {SENSOR_CSV}. Did you run prep_cmapss.py?")
    
    sensor_agent = SensorAgent(sensor_csv_path=SENSOR_CSV)
    historical_df = sensor_agent.get_historical_data()

    # 2. Train and Save Analysis Agent (Isolation Forest)
    print("\n[2/3] Training Anomaly Detector...")
    analysis_agent = AnalysisAgent()
    analysis_agent.train(historical_df)
    analysis_agent.save_model("model_analysis.pkl")

    # 3. Train and Save Diagnosis Agent (Random Forest)
    print("\n[3/3] Training Fault Classifier...")
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}. Did you run prep_cmapss.py?")
        
    diagnosis_agent = DiagnosisAgent()
    diagnosis_agent.train(labels_csv_path=LABELS_CSV)
    diagnosis_agent.save_model("model_diagnosis.pkl")

    print("\n=========================================")
    print("  ✅ Training Complete! Models saved.")
    print("  Run 'python orchestrator_agent.py' to use them.")
    print("=========================================\n")

if __name__ == "__main__":
    train()
