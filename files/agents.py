"""
Predictive Maintenance using Agentic AI
agents.py - Core AI Agents

Architecture:
  SensorAgent     -> Collects & monitors sensor data
  AnalysisAgent   -> Detects anomalies using Isolation Forest + LSTM features
  DiagnosisAgent  -> Diagnoses fault type & severity
  MaintenanceAgent-> Schedules maintenance actions
  OrchestratorAgent -> Coordinates all agents
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
import random
import json


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class SensorReading:
    machine_id: str
    timestamp: datetime
    temperature: float       # °C
    vibration: float         # mm/s
    pressure: float          # bar
    rpm: float               # rotations per minute
    current: float           # amperes
    oil_level: float         # percentage

@dataclass
class AnomalyReport:
    machine_id: str
    timestamp: datetime
    anomaly_score: float
    is_anomalous: bool
    features_flagged: list[str]

@dataclass
class DiagnosisReport:
    machine_id: str
    timestamp: datetime
    fault_type: str
    severity: str            # LOW / MEDIUM / HIGH / CRITICAL
    confidence: float
    recommended_action: str
    estimated_rul: int       # Remaining Useful Life in hours

@dataclass
class MaintenanceTask:
    machine_id: str
    created_at: datetime
    scheduled_for: datetime
    task_type: str
    priority: str
    technician_notes: str
    estimated_downtime_hours: float


# ─────────────────────────────────────────────
# SENSOR AGENT
# ─────────────────────────────────────────────

class SensorAgent:
    """
    Simulates and collects sensor readings from industrial machines.
    In production, this would interface with OPC-UA / MQTT / REST APIs.
    """

    MACHINES = ["MACH-001", "MACH-002", "MACH-003", "MACH-004", "MACH-005"]

    # Baseline healthy ranges
    HEALTHY_RANGES = {
        "temperature": (60, 85),
        "vibration":   (0.5, 3.0),
        "pressure":    (4.0, 7.0),
        "rpm":         (1400, 1600),
        "current":     (8.0, 12.0),
        "oil_level":   (70, 100),
    }

    def __init__(self):
        self.fault_probabilities = {m: random.uniform(0, 0.3) for m in self.MACHINES}
        self.degradation_state = {m: random.uniform(0, 0.4) for m in self.MACHINES}
        print(f"[SensorAgent] Initialized. Monitoring {len(self.MACHINES)} machines.")

    def _generate_reading(self, machine_id: str, timestamp: datetime) -> SensorReading:
        deg = self.degradation_state[machine_id]
        fp  = self.fault_probabilities[machine_id]

        def sample(key, extra_noise=0):
            lo, hi = self.HEALTHY_RANGES[key]
            center = (lo + hi) / 2
            base = np.random.normal(center, (hi - lo) * 0.08)
            fault_shift = deg * (hi - lo) * random.uniform(0, 1.5) * fp
            noise = np.random.normal(0, extra_noise)
            return round(float(np.clip(base + fault_shift + noise, lo * 0.7, hi * 1.4)), 3)

        return SensorReading(
            machine_id=machine_id,
            timestamp=timestamp,
            temperature=sample("temperature", 1.0) + deg * 20 * fp,
            vibration=sample("vibration", 0.1) + deg * 4 * fp,
            pressure=sample("pressure", 0.1) - deg * 1.5 * fp,
            rpm=sample("rpm", 20) - deg * 100 * fp,
            current=sample("current", 0.3) + deg * 3 * fp,
            oil_level=max(10, sample("oil_level", 2) - deg * 30 * fp),
        )

    def collect_readings(self, n_steps: int = 1, interval_minutes: int = 5) -> list[SensorReading]:
        readings = []
        now = datetime.now()
        for machine_id in self.MACHINES:
            # Slowly degrade machines over time
            self.degradation_state[machine_id] = min(
                1.0, self.degradation_state[machine_id] + random.uniform(0, 0.02)
            )
            for step in range(n_steps):
                ts = now - timedelta(minutes=interval_minutes * (n_steps - step))
                readings.append(self._generate_reading(machine_id, ts))
        return readings

    def get_historical_data(self, hours: int = 48) -> pd.DataFrame:
        """Generate synthetic historical sensor data for training."""
        records = []
        now = datetime.now()
        for machine_id in self.MACHINES:
            for h in range(hours * 12):  # every 5 minutes
                ts = now - timedelta(minutes=5 * h)
                r = self._generate_reading(machine_id, ts)
                records.append({
                    "machine_id": r.machine_id,
                    "timestamp": r.timestamp,
                    "temperature": r.temperature,
                    "vibration": r.vibration,
                    "pressure": r.pressure,
                    "rpm": r.rpm,
                    "current": r.current,
                    "oil_level": r.oil_level,
                })
        return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────
# ANALYSIS AGENT
# ─────────────────────────────────────────────

class AnalysisAgent:
    """
    Detects anomalies in sensor streams using Isolation Forest.
    Trained on historical healthy data; flags statistical outliers.
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
        print(f"[AnalysisAgent] Model trained on {len(X):,} historical samples.")

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
        preds = self.model.predict(X)  # -1 = anomaly, 1 = normal

        reports = []
        for i, r in enumerate(readings):
            anomaly_score = float(np.clip(-scores[i], 0, 1))
            is_anomalous = preds[i] == -1

            # Identify which features are out of range
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


# ─────────────────────────────────────────────
# DIAGNOSIS AGENT
# ─────────────────────────────────────────────

class DiagnosisAgent:
    """
    Classifies fault type and estimates remaining useful life (RUL).
    Uses Random Forest trained on labeled fault patterns.
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
        print("[DiagnosisAgent] Initialized. Fault classifier: Random Forest.")

    def _generate_training_data(self) -> tuple[np.ndarray, list[str]]:
        """Synthesize labeled fault data for training."""
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

    def train(self):
        X, y = self._generate_training_data()
        self.model.fit(X, y)
        self.is_trained = True
        print(f"[DiagnosisAgent] Fault classifier trained on {len(y):,} labeled samples.")

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

        # Severity based on anomaly score + confidence
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

        # Estimate RUL (heuristic: inverse of anomaly score)
        base_rul = 500
        rul = max(10, int(base_rul * (1 - anomaly.anomaly_score) * random.uniform(0.8, 1.2)))

        return DiagnosisReport(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=round(confidence, 3),
            recommended_action=self.ACTIONS[fault_type],
            estimated_rul=rul,
        )


# ─────────────────────────────────────────────
# MAINTENANCE AGENT
# ─────────────────────────────────────────────

class MaintenanceAgent:
    """
    Plans and schedules maintenance tasks based on diagnosis.
    Applies priority-based scheduling logic.
    """

    def __init__(self):
        self.task_queue: list[MaintenanceTask] = []
        self.completed_tasks: list[MaintenanceTask] = []
        print("[MaintenanceAgent] Initialized. Task scheduler ready.")

    def schedule(self, diagnosis: DiagnosisReport) -> Optional[MaintenanceTask]:
        if diagnosis.fault_type == "Normal Operation" and diagnosis.severity == "LOW":
            return None

        delay_hours = {
            "CRITICAL": 2,
            "HIGH":     24,
            "MEDIUM":   72,
            "LOW":      168,
        }

        downtime = {
            "CRITICAL": 8.0,
            "HIGH":     4.0,
            "MEDIUM":   2.0,
            "LOW":      1.0,
        }

        task = MaintenanceTask(
            machine_id=diagnosis.machine_id,
            created_at=diagnosis.timestamp,
            scheduled_for=diagnosis.timestamp + timedelta(hours=delay_hours[diagnosis.severity]),
            task_type=diagnosis.fault_type,
            priority=diagnosis.severity,
            technician_notes=(
                f"Fault: {diagnosis.fault_type} | "
                f"Confidence: {diagnosis.confidence:.1%} | "
                f"Est. RUL: {diagnosis.estimated_rul}h | "
                f"Action: {diagnosis.recommended_action}"
            ),
            estimated_downtime_hours=downtime[diagnosis.severity],
        )
        self.task_queue.append(task)
        return task

    def get_prioritized_queue(self) -> list[MaintenanceTask]:
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        return sorted(self.task_queue, key=lambda t: (priority_order[t.priority], t.scheduled_for))


# ─────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────

class OrchestratorAgent:
    """
    Master agent that coordinates the full predictive maintenance pipeline:
    Sensor -> Analysis -> Diagnosis -> Maintenance Scheduling
    """

    def __init__(self):
        print("\n" + "="*60)
        print("  Predictive Maintenance — Agentic AI System")
        print("="*60)
        self.sensor_agent      = SensorAgent()
        self.analysis_agent    = AnalysisAgent()
        self.diagnosis_agent   = DiagnosisAgent()
        self.maintenance_agent = MaintenanceAgent()

        self.all_readings:    list[SensorReading]  = []
        self.all_anomalies:   list[AnomalyReport]  = []
        self.all_diagnoses:   list[DiagnosisReport] = []
        self.all_tasks:       list[MaintenanceTask] = []

    def bootstrap(self):
        """Train all ML agents on historical data."""
        print("\n[Orchestrator] Phase 1: Bootstrapping agents...")
        historical = self.sensor_agent.get_historical_data(hours=48)
        self.analysis_agent.train(historical)
        self.diagnosis_agent.train()
        print("[Orchestrator] All agents ready.\n")

    def run_cycle(self, n_steps: int = 3) -> dict:
        """Execute one full monitoring cycle."""
        print(f"[Orchestrator] Running monitoring cycle ({n_steps} timesteps per machine)...")

        readings  = self.sensor_agent.collect_readings(n_steps=n_steps)
        anomalies = self.analysis_agent.analyze(readings)

        tasks_created = []
        diagnoses = []
        for reading, anomaly in zip(readings, anomalies):
            diagnosis = self.diagnosis_agent.diagnose(reading, anomaly)
            diagnoses.append(diagnosis)
            if anomaly.is_anomalous or diagnosis.severity in ("HIGH", "CRITICAL"):
                task = self.maintenance_agent.schedule(diagnosis)
                if task:
                    tasks_created.append(task)

        self.all_readings.extend(readings)
        self.all_anomalies.extend(anomalies)
        self.all_diagnoses.extend(diagnoses)
        self.all_tasks.extend(tasks_created)

        n_anomalous = sum(1 for a in anomalies if a.is_anomalous)
        print(f"  Readings: {len(readings)} | Anomalies: {n_anomalous} | Tasks Created: {len(tasks_created)}")

        return {
            "readings":   readings,
            "anomalies":  anomalies,
            "diagnoses":  diagnoses,
            "tasks":      tasks_created,
        }

    def get_system_summary(self) -> dict:
        """Return high-level system health summary."""
        machines = SensorAgent.MACHINES
        summary = {}
        for mid in machines:
            machine_diagnoses = [d for d in self.all_diagnoses if d.machine_id == mid]
            machine_anomalies = [a for a in self.all_anomalies if a.machine_id == mid]
            if not machine_diagnoses:
                continue
            latest = machine_diagnoses[-1]
            total  = len(machine_anomalies)
            flagged = sum(1 for a in machine_anomalies if a.is_anomalous)
            summary[mid] = {
                "latest_fault":    latest.fault_type,
                "severity":        latest.severity,
                "estimated_rul":   latest.estimated_rul,
                "anomaly_rate":    round(flagged / total, 3) if total else 0,
                "health_score":    round(max(0, 100 - (machine_anomalies[-1].anomaly_score * 100 if machine_anomalies else 20)), 1),
            }
        return summary


if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    orchestrator.bootstrap()

    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        result = orchestrator.run_cycle(n_steps=2)

    print("\n=== MAINTENANCE QUEUE ===")
    for task in orchestrator.maintenance_agent.get_prioritized_queue():
        print(f"  [{task.priority}] {task.machine_id} | {task.task_type} | "
              f"Scheduled: {task.scheduled_for.strftime('%Y-%m-%d %H:%M')}")

    print("\n=== SYSTEM SUMMARY ===")
    for mid, info in orchestrator.get_system_summary().items():
        print(f"  {mid}: {info['latest_fault']} | Severity: {info['severity']} | RUL: {info['estimated_rul']}h")
