"""
Predictive Maintenance — Data Loader
data_loader.py — CSV / Real Data Ingestion

Handles loading sensor readings and labeled fault data from CSV files.
Validates schema, parses timestamps, and converts rows to agent-compatible formats.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# REQUIRED COLUMNS
# ─────────────────────────────────────────────

SENSOR_COLUMNS = [
    "machine_id", "timestamp",
    "temperature", "vibration", "pressure", "rpm", "current", "oil_level",
]

FAULT_LABEL_COLUMNS = [
    "temperature", "vibration", "pressure", "rpm", "current", "oil_level", "fault_type",
]


# ─────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, required: list[str], file_label: str = "CSV") -> list[str]:
    """
    Check that a DataFrame contains all required columns.
    Returns list of missing column names (empty if all present).
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"[DataLoader] {file_label} is missing required columns: {missing}\n"
            f"  Found columns: {list(df.columns)}\n"
            f"  Required: {required}"
        )
    return missing


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Try to parse the 'timestamp' column into datetime objects."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = df["timestamp"].isna().sum()
        if bad > 0:
            print(f"[DataLoader] Warning: {bad} rows had unparseable timestamps and were dropped.")
            df = df.dropna(subset=["timestamp"])
    return df


# ─────────────────────────────────────────────
# SENSOR DATA LOADER
# ─────────────────────────────────────────────

def load_sensor_csv(path: str | Path) -> pd.DataFrame:
    """
    Load sensor readings from a CSV file.

    Expected columns:
        machine_id, timestamp, temperature, vibration, pressure, rpm, current, oil_level

    Returns a cleaned, sorted DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[DataLoader] Sensor CSV not found: {path}")

    print(f"[DataLoader] Loading sensor data from: {path}")
    df = pd.read_csv(path)
    validate_columns(df, SENSOR_COLUMNS, file_label="Sensor CSV")

    # Parse timestamps
    df = _parse_timestamps(df)

    # Ensure numeric types for sensor columns
    numeric_cols = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any NaN sensor values
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    if before != after:
        print(f"[DataLoader] Dropped {before - after} rows with invalid sensor values.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    machines = df["machine_id"].nunique()
    print(f"[DataLoader] Loaded {len(df):,} readings from {machines} machine(s).")
    return df


# ─────────────────────────────────────────────
# FAULT LABELS LOADER
# ─────────────────────────────────────────────

def load_fault_labels_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load labeled fault data for training the DiagnosisAgent.

    Expected columns:
        temperature, vibration, pressure, rpm, current, oil_level, fault_type

    Returns (X, y) where X is a numpy array of features and y is a list of labels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[DataLoader] Fault labels CSV not found: {path}")

    print(f"[DataLoader] Loading fault labels from: {path}")
    df = pd.read_csv(path)
    validate_columns(df, FAULT_LABEL_COLUMNS, file_label="Fault Labels CSV")

    # Ensure numeric features
    feature_cols = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols + ["fault_type"])

    X = df[feature_cols].values
    y = df["fault_type"].tolist()

    unique_faults = set(y)
    print(f"[DataLoader] Loaded {len(y):,} labeled samples across {len(unique_faults)} fault types: {unique_faults}")
    return X, y


# ─────────────────────────────────────────────
# SENSOR DF → SensorReading CONVERSION
# ─────────────────────────────────────────────

def df_to_sensor_readings(df: pd.DataFrame):
    """
    Convert a sensor DataFrame into a list of SensorReading dataclass objects.
    Import is done locally to avoid circular imports.
    """
    from agents import SensorReading

    readings = []
    for _, row in df.iterrows():
        readings.append(SensorReading(
            machine_id=str(row["machine_id"]),
            timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
            temperature=float(row["temperature"]),
            vibration=float(row["vibration"]),
            pressure=float(row["pressure"]),
            rpm=float(row["rpm"]),
            current=float(row["current"]),
            oil_level=float(row["oil_level"]),
        ))
    return readings


# ─────────────────────────────────────────────
# CONVENIENCE: Load from uploaded Streamlit file
# ─────────────────────────────────────────────

def load_sensor_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load sensor CSV from a Streamlit UploadedFile object."""
    df = pd.read_csv(uploaded_file)
    validate_columns(df, SENSOR_COLUMNS, file_label="Uploaded Sensor CSV")
    df = _parse_timestamps(df)

    numeric_cols = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[DataLoader] Uploaded sensor CSV: {len(df):,} readings, {df['machine_id'].nunique()} machine(s).")
    return df


def load_faults_from_uploaded_file(uploaded_file) -> tuple[np.ndarray, list[str]]:
    """Load fault labels CSV from a Streamlit UploadedFile object."""
    df = pd.read_csv(uploaded_file)
    validate_columns(df, FAULT_LABEL_COLUMNS, file_label="Uploaded Fault Labels CSV")

    feature_cols = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["fault_type"])

    X = df[feature_cols].values
    y = df["fault_type"].tolist()
    print(f"[DataLoader] Uploaded fault labels: {len(y):,} samples.")
    return X, y
