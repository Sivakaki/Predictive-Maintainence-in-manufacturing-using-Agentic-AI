import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Load CMAPSS FD001
print("Loading C-MAPSS train_FD001.txt...")
df = pd.read_csv('CMaps/train_FD001.txt', sep=r'\s+', header=None)

# 2. Rename columns
# Columns: 0: unit/machine_id, 1: cycle
# We will pick 6 sensors from the 21 sensors to represent our expected fields.
# Sensor 2 (col 6), 3 (col 7), 4 (col 8), 7 (col 11), 8 (col 12), 11 (col 15)
df = df[[0, 1, 6, 7, 8, 11, 12, 15]]
df.columns = ["machine_id", "cycle", "s1", "s2", "s3", "s4", "s5", "s6"]

df["machine_id"] = "MACH-" + df["machine_id"].astype(str).str.zfill(3)

# 3. Create RUL (Remaining Useful Life) column to label faults
max_cycles = df.groupby("machine_id")["cycle"].max()
df["rul"] = df.apply(lambda row: max_cycles[row["machine_id"]] - row["cycle"], axis=1)

# 4. Map scaled values to our 6 features to be compatible with the Dashboard
EXPECTED_RANGES = {
    "temperature": (60, 85),
    "vibration":   (0.5, 3.0),
    "pressure":    (4.0, 7.0),
    "rpm":         (1400, 1600),
    "current":     (8.0, 12.0),
    "oil_level":   (70, 100),
}

def minmax_scale(series, range_min, range_max):
    s_min, s_max = series.min(), series.max()
    # Handle constant sensors to avoid division by zero
    if s_min == s_max:
        return pd.Series(np.full(len(series), (range_min + range_max)/2), index=series.index)
    
    scaled = (series - s_min) / (s_max - s_min) # 0 to 1
    # For some metrics, as engine degrades, value goes up. We stretch them to our ranges.
    # To introduce realistic variance and anomalies crossing the threshold, we scale to 0.7 * min and 1.3 * max sometimes,
    # but let's just stick to a slightly inflated range to ensure faults exceed healthy thresholds automatically.
    return range_min + scaled * (range_max - range_min) * 1.5

df["temperature"] = minmax_scale(df["s1"], *EXPECTED_RANGES["temperature"])
df["vibration"]   = minmax_scale(df["s2"], *EXPECTED_RANGES["vibration"])
df["pressure"]    = minmax_scale(df["s3"], *EXPECTED_RANGES["pressure"])
df["rpm"]         = minmax_scale(df["s4"], *EXPECTED_RANGES["rpm"])
df["current"]     = minmax_scale(df["s5"], *EXPECTED_RANGES["current"])
# Reverse scaled oil_level as oil level usually goes down when bad
df["oil_level"]   = EXPECTED_RANGES["oil_level"][1] - minmax_scale(df["s6"], 0, (EXPECTED_RANGES["oil_level"][1] - EXPECTED_RANGES["oil_level"][0]) * 1.5)

# 5. Add timestamps based on cycles
start_time = datetime(2024, 6, 1, 8, 0, 0)
df["timestamp"] = df.apply(lambda row: start_time + timedelta(hours=row["cycle"]), axis=1)

# 6. Prepare Sensor Data CSV (Just take the first 3 machines to keep dashboard fast, all cycles)
subset_machines = ["MACH-001", "MACH-002", "MACH-003"]
sensor_df = df[df["machine_id"].isin(subset_machines)].copy()
sensor_csv_cols = ["machine_id", "timestamp", "temperature", "vibration", "pressure", "rpm", "current", "oil_level"]
sensor_df[sensor_csv_cols].to_csv("cmaps_sensor_data.csv", index=False)
print(f"Saved cmaps_sensor_data.csv with {len(sensor_df)} rows.")

# 7. Prepare Fault Labels CSV (Use all machines for training data)
# Map faults based on RUL
def assign_fault(rul):
    if rul > 100:
        return "Normal Operation"
    elif rul > 50:
        return "Pressure Drop"
    elif rul > 20:
        return "Overheating"
    else:
        return "Bearing Wear"

df["fault_type"] = df["rul"].apply(assign_fault)
labels_csv_cols = ["temperature", "vibration", "pressure", "rpm", "current", "oil_level", "fault_type"]

# To avoid massive file size and class imbalance, sample 100 from each fault type
labels_df = df.groupby("fault_type").apply(lambda x: x.sample(n=min(200, len(x)), random_state=42)).reset_index(drop=True)
labels_df[labels_csv_cols].to_csv("cmaps_fault_labels.csv", index=False)
print(f"Saved cmaps_fault_labels.csv with {len(labels_df)} rows for training.")
