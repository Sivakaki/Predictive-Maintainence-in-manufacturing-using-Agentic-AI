"""
Predictive Maintenance — Data Models
models.py — Shared data classes used across all agents
"""

from dataclasses import dataclass
from datetime import datetime


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
