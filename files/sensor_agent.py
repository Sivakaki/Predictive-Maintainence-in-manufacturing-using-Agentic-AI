"""
Predictive Maintenance — Sensor Agent
sensor_agent.py — Collects & monitors sensor data (CSV or simulated)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Optional

from models import SensorReading


class SensorAgent:
    """
    Collects sensor readings from industrial machines.
    Supports two modes:
      - Real data:  load from CSV files via data_loader module
      - Simulated:  generate synthetic readings (fallback / demo mode)
    """

    MACHINES = ["MACH-001", "MACH-002", "MACH-003", "MACH-004", "MACH-005"]

    # Baseline healthy ranges (used for anomaly flagging)
    HEALTHY_RANGES = {
        "temperature": (60, 85),
        "vibration":   (0.5, 3.0),
        "pressure":    (4.0, 7.0),
        "rpm":         (1400, 1600),
        "current":     (8.0, 12.0),
        "oil_level":   (70, 100),
    }

    def __init__(self, sensor_csv_path: Optional[str] = None):
        """
        Args:
            sensor_csv_path: Path to a CSV file with real sensor data.
                             If None, the agent runs in simulated mode.
        """
        self.sensor_csv_path = sensor_csv_path
        self._csv_data: Optional[pd.DataFrame] = None
        self._csv_cursor: int = 0

        if sensor_csv_path:
            from data_loader import load_sensor_csv
            self._csv_data = load_sensor_csv(sensor_csv_path)
            self.MACHINES = sorted(self._csv_data["machine_id"].unique().tolist())
            self.data_mode = "CSV"
            print(f"[SensorAgent] Initialized in CSV mode. {len(self.MACHINES)} machine(s) from file.")
        else:
            self.data_mode = "SIMULATED"
            self.fault_probabilities = {m: random.uniform(0, 0.3) for m in self.MACHINES}
            self.degradation_state = {m: random.uniform(0, 0.4) for m in self.MACHINES}
            print(f"[SensorAgent] Initialized in SIMULATED mode. Monitoring {len(self.MACHINES)} machines.")

    # ── CSV MODE METHODS ──────────────────────

    def load_readings_from_csv(self) -> list[SensorReading]:
        """Load ALL readings from the CSV as SensorReading objects."""
        if self._csv_data is None:
            raise RuntimeError("No CSV data loaded. Provide sensor_csv_path.")
        from data_loader import df_to_sensor_readings
        return df_to_sensor_readings(self._csv_data)

    def load_readings_from_dataframe(self, df: pd.DataFrame) -> list[SensorReading]:
        """Load readings from an in-memory DataFrame (e.g. from Streamlit upload)."""
        self._csv_data = df
        self.MACHINES = sorted(df["machine_id"].unique().tolist())
        self.data_mode = "CSV"
        self._csv_cursor = 0
        from data_loader import df_to_sensor_readings
        return df_to_sensor_readings(df)

    def get_historical_data_from_csv(self) -> pd.DataFrame:
        """Return the full CSV DataFrame for model training."""
        if self._csv_data is None:
            raise RuntimeError("No CSV data loaded.")
        return self._csv_data.copy()

    def collect_readings_csv(self, batch_size: int = 10) -> list[SensorReading]:
        """
        Consume the next batch of readings from CSV.
        Simulates a streaming pipeline by advancing a cursor.
        """
        if self._csv_data is None:
            raise RuntimeError("No CSV data loaded.")
        from data_loader import df_to_sensor_readings

        total = len(self._csv_data)
        start = self._csv_cursor
        end = min(start + batch_size, total)

        if start >= total:
            self._csv_cursor = 0
            start = 0
            end = min(batch_size, total)

        batch_df = self._csv_data.iloc[start:end]
        self._csv_cursor = end
        return df_to_sensor_readings(batch_df)

    # ── SIMULATED MODE METHODS ────────────────

    def _generate_reading(self, machine_id: str, timestamp: datetime) -> SensorReading:
        """Generate a single simulated sensor reading."""
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

    def collect_readings_simulated(self, n_steps: int = 1, interval_minutes: int = 5) -> list[SensorReading]:
        """Generate simulated sensor readings."""
        readings = []
        now = datetime.now()
        for machine_id in self.MACHINES:
            self.degradation_state[machine_id] = min(
                1.0, self.degradation_state[machine_id] + random.uniform(0, 0.02)
            )
            for step in range(n_steps):
                ts = now - timedelta(minutes=interval_minutes * (n_steps - step))
                readings.append(self._generate_reading(machine_id, ts))
        return readings

    def get_historical_data_simulated(self, hours: int = 48) -> pd.DataFrame:
        """Generate synthetic historical sensor data for training."""
        records = []
        now = datetime.now()
        for machine_id in self.MACHINES:
            for h in range(hours * 12):
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

    # ── UNIFIED INTERFACE ─────────────────────

    def collect_readings(self, n_steps: int = 1, interval_minutes: int = 5) -> list[SensorReading]:
        """Collect readings using the active data source."""
        if self.data_mode == "CSV":
            batch_size = n_steps * len(self.MACHINES)
            return self.collect_readings_csv(batch_size=batch_size)
        else:
            return self.collect_readings_simulated(n_steps, interval_minutes)

    def get_historical_data(self, hours: int = 48) -> pd.DataFrame:
        """Get historical data for model training."""
        if self.data_mode == "CSV":
            return self.get_historical_data_from_csv()
        else:
            return self.get_historical_data_simulated(hours)
