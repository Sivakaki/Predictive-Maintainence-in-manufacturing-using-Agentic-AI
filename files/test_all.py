"""Quick verification script to test all agents work correctly."""

print("--- Testing individual imports ---")
from models import SensorReading, AnomalyReport, DiagnosisReport, MaintenanceTask
print("[OK] models.py")

from sensor_agent import SensorAgent
print("[OK] sensor_agent.py")

from analysis_agent import AnalysisAgent
print("[OK] analysis_agent.py")

from diagnosis_agent import DiagnosisAgent
print("[OK] diagnosis_agent.py")

from maintenance_agent import MaintenanceAgent
print("[OK] maintenance_agent.py")

from orchestrator_agent import OrchestratorAgent
print("[OK] orchestrator_agent.py")

from data_loader import load_sensor_csv, load_fault_labels_csv
print("[OK] data_loader.py")

from agents import SensorAgent, OrchestratorAgent
print("[OK] agents.py (re-exports)")

print()
print("--- Testing SIMULATED pipeline ---")
o = OrchestratorAgent()
o.bootstrap()
r = o.run_cycle(n_steps=1)
print(f"  Readings: {len(r['readings'])}")
anomaly_count = sum(1 for a in r["anomalies"] if a.is_anomalous)
print(f"  Anomalies: {anomaly_count}")
print(f"  Diagnoses: {len(r['diagnoses'])}")

print()
print("--- Testing CSV pipeline ---")
o2 = OrchestratorAgent(sensor_csv_path="sample_sensor_data.csv", labels_csv_path="sample_fault_labels.csv")
o2.bootstrap()
r2 = o2.run_cycle(n_steps=1)
print(f"  Readings: {len(r2['readings'])}")
machines = sorted(set(rd.machine_id for rd in r2["readings"]))
print(f"  Machines: {machines}")

print()
print("=" * 40)
print("  ALL TESTS PASSED")
print("=" * 40)
