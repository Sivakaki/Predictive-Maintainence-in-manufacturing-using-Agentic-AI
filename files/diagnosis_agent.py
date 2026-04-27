"""
Predictive Maintenance — Diagnosis Agent
diagnosis_agent.py — Classifies fault type & severity using Random Forest
"""

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional
import joblib
import os

from models import SensorReading, AnomalyReport, DiagnosisReport


class DiagnosisAgent:
    """
    Classifies fault type and estimates remaining useful life (RUL).
    Supports training from:
      - Real labeled CSV data (via data_loader)
      - Synthetic fault patterns (fallback)
    """

    FAULT_TYPES = [
        "Bearing Wear",
        "Overheating",
        "Lubrication Failure",
        "Imbalance/Misalignment",
        "Electrical Fault",
        "Pressure Drop",
        "Normal Operation",
    ]

    ACTIONS = {
        "Bearing Wear":            "Schedule bearing inspection & replacement",
        "Overheating":             "Check cooling system and reduce load",
        "Lubrication Failure":     "Immediate oil top-up and filter check",
        "Imbalance/Misalignment":  "Balance check and shaft alignment",
        "Electrical Fault":        "Electrical diagnostic by certified technician",
        "Pressure Drop":           "Inspect seals, valves, and piping",
        "Normal Operation":        "Continue routine monitoring",
    }

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
        ])
        self.is_trained = False
        self.training_source = None
        print("[DiagnosisAgent] Initialized. Fault classifier: Random Forest.")

    def _generate_training_data(self) -> tuple[np.ndarray, list[str]]:
        """Synthesize labeled fault data for training (fallback mode)."""
        X, y = [], []
        rng = np.random.default_rng(42)

        patterns = {
            "Normal Operation":        [72, 1.5, 5.5, 1500, 10.0, 85],
            "Bearing Wear":            [80, 6.5, 5.4, 1480, 11.5, 82],
            "Overheating":             [105, 2.0, 5.3, 1490, 13.5, 78],
            "Lubrication Failure":     [92, 4.0, 5.1, 1460, 12.0, 35],
            "Imbalance/Misalignment":  [76, 8.0, 5.5, 1520, 10.5, 80],
            "Electrical Fault":        [78, 1.8, 5.4, 1350, 16.0, 83],
            "Pressure Drop":           [74, 1.6, 2.5, 1490, 10.2, 81],
        }
        noise_scale = [3, 0.8, 0.3, 30, 0.6, 5]

        for fault, center in patterns.items():
            n = 300
            noise = rng.normal(0, noise_scale, (n, 6))
            samples = np.array(center) + noise
            X.append(samples)
            y.extend([fault] * n)

        return np.vstack(X), y

    def train(self, labels_csv_path: Optional[str] = None):
        """
        Train the fault classifier.
        If labels_csv_path is provided, loads real labeled data.
        Otherwise falls back to synthetic training data.
        """
        if labels_csv_path:
            from data_loader import load_fault_labels_csv
            X, y = load_fault_labels_csv(labels_csv_path)
            self.training_source = "CSV"
        else:
            X, y = self._generate_training_data()
            self.training_source = "SYNTHETIC"

        self.model.fit(X, y)
        self.is_trained = True

        unique_faults = set(y)
        for ft in unique_faults:
            if ft not in self.ACTIONS:
                self.ACTIONS[ft] = f"Investigate {ft} — no predefined action"
            if ft not in self.FAULT_TYPES:
                self.FAULT_TYPES.append(ft)

        print(f"[DiagnosisAgent] Classifier trained on {len(y):,} samples ({self.training_source}).")

    def train_from_data(self, X: np.ndarray, y: list[str]):
        """Train directly from in-memory data (e.g. from Streamlit upload)."""
        self.model.fit(X, y)
        self.is_trained = True
        self.training_source = "UPLOADED CSV"

        unique_faults = set(y)
        for ft in unique_faults:
            if ft not in self.ACTIONS:
                self.ACTIONS[ft] = f"Investigate {ft} — no predefined action"
            if ft not in self.FAULT_TYPES:
                self.FAULT_TYPES.append(ft)

        print(f"[DiagnosisAgent] Classifier trained on {len(y):,} uploaded samples.")

    def save_model(self, model_path: str = "model_diagnosis.pkl"):
        """Save the trained model and config to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        state = {
            "model": self.model,
            "actions": self.ACTIONS,
            "fault_types": self.FAULT_TYPES
        }
        joblib.dump(state, model_path)
        print(f"[DiagnosisAgent] Model saved precisely to {model_path}")

    def load_model(self, model_path: str = "model_diagnosis.pkl") -> bool:
        """Load a trained model from disk if it exists. Returns True if successful."""
        if os.path.exists(model_path):
            state = joblib.load(model_path)
            self.model = state["model"]
            self.ACTIONS = state["actions"]
            self.FAULT_TYPES = state["fault_types"]
            self.is_trained = True
            print(f"[DiagnosisAgent] Model loaded from {model_path}")
            return True
        return False

    def diagnose(self, reading: SensorReading, anomaly: AnomalyReport) -> DiagnosisReport:
        if not self.is_trained:
            raise RuntimeError("DiagnosisAgent must be trained first.")

        features = np.array([[
            reading.temperature, reading.vibration, reading.pressure,
            reading.rpm, reading.current, reading.oil_level
        ]])

        fault_type = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = float(np.max(proba))

        score = anomaly.anomaly_score
        if fault_type == "Normal Operation":
            severity = "LOW"
        elif score > 0.7 or confidence > 0.85:
            severity = "CRITICAL"
        elif score > 0.5:
            severity = "HIGH"
        elif score > 0.3:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        base_rul = 500
        rul = max(10, int(base_rul * (1 - anomaly.anomaly_score) * random.uniform(0.8, 1.2)))

        return DiagnosisReport(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=round(confidence, 3),
            recommended_action=self.ACTIONS.get(fault_type, "Investigate further"),
            estimated_rul=rul,
        )
