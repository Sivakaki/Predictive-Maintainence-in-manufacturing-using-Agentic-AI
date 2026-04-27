"""
Predictive Maintenance — Analysis Agent
analysis_agent.py — Detects anomalies using Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

from models import SensorReading, AnomalyReport
from sensor_agent import SensorAgent


class AnalysisAgent:
    """
    Detects anomalies in sensor streams using Isolation Forest.
    Trained on historical data (real or simulated); flags statistical outliers.
    """

    FEATURES = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level"]

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42,
                n_jobs=-1
            ))
        ])
        self.is_trained = False
        print("[AnalysisAgent] Initialized. Anomaly detection: Isolation Forest.")

    def train(self, historical_df: pd.DataFrame):
        X = historical_df[self.FEATURES].dropna()
        self.model.fit(X)
        self.is_trained = True
        print(f"[AnalysisAgent] Model trained on {len(X):,} samples.")

    def save_model(self, model_path: str = "model_analysis.pkl"):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self.model, model_path)
        print(f"[AnalysisAgent] Model saved precisely to {model_path}")

    def load_model(self, model_path: str = "model_analysis.pkl") -> bool:
        """Load a trained model from disk if it exists. Returns True if successful."""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
            print(f"[AnalysisAgent] Model loaded from {model_path}")
            return True
        return False

    def analyze(self, readings: list[SensorReading]) -> list[AnomalyReport]:
        if not self.is_trained:
            raise RuntimeError("AnalysisAgent must be trained before analyzing.")

        rows = [{f: getattr(r, f) for f in self.FEATURES} | {"machine_id": r.machine_id, "timestamp": r.timestamp}
                for r in readings]
        df = pd.DataFrame(rows)
        X = df[self.FEATURES]

        scores = self.model.named_steps["iforest"].decision_function(
            self.model.named_steps["scaler"].transform(X)
        )
        preds = self.model.predict(X)

        reports = []
        for i, r in enumerate(readings):
            anomaly_score = float(np.clip(-scores[i], 0, 1))
            is_anomalous = preds[i] == -1

            flagged = []
            for feat in self.FEATURES:
                val = getattr(r, feat)
                lo, hi = SensorAgent.HEALTHY_RANGES[feat]
                if val < lo * 0.85 or val > hi * 1.15:
                    flagged.append(feat)

            reports.append(AnomalyReport(
                machine_id=r.machine_id,
                timestamp=r.timestamp,
                anomaly_score=round(anomaly_score, 4),
                is_anomalous=is_anomalous or len(flagged) > 1,
                features_flagged=flagged,
            ))
        return reports
