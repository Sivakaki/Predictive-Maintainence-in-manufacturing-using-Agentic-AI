"""
Predictive Maintenance — Orchestrator Agent
orchestrator_agent.py — Coordinates all agents in the pipeline
"""

from typing import Optional

from models import SensorReading, AnomalyReport, DiagnosisReport, MaintenanceTask
from sensor_agent import SensorAgent
from analysis_agent import AnalysisAgent
from diagnosis_agent import DiagnosisAgent
from maintenance_agent import MaintenanceAgent
from inventory_agent import InventoryAgent


class OrchestratorAgent:
    """
    Master agent that coordinates the full predictive maintenance pipeline:
    Sensor -> Analysis -> Diagnosis -> Maintenance Scheduling

    Supports both CSV-based real data and simulated data.
    """

    def __init__(self, sensor_csv_path: Optional[str] = None, labels_csv_path: Optional[str] = None):
        """
        Args:
            sensor_csv_path:  Path to sensor readings CSV. None = simulated.
            labels_csv_path:  Path to labeled fault data CSV. None = synthetic training.
        """
        print("\n" + "="*60)
        print("  Predictive Maintenance — Agentic AI System")
        print("="*60)

        self.sensor_csv_path = sensor_csv_path
        self.labels_csv_path = labels_csv_path

        self.sensor_agent      = SensorAgent(sensor_csv_path=sensor_csv_path)
        self.analysis_agent    = AnalysisAgent()
        self.diagnosis_agent   = DiagnosisAgent()
        self.inventory_agent   = InventoryAgent()
        self.maintenance_agent = MaintenanceAgent(inventory_agent=self.inventory_agent)

        self.all_readings:    list[SensorReading]  = []
        self.all_anomalies:   list[AnomalyReport]  = []
        self.all_diagnoses:   list[DiagnosisReport] = []
        self.all_tasks:       list[MaintenanceTask] = []

        data_mode = "CSV" if sensor_csv_path else "SIMULATED"
        print(f"[Orchestrator] Data mode: {data_mode}")

    def bootstrap(self, force_train: bool = False):
        """
        Initialize agents. If models exist on disk, load them.
        Otherwise, train all ML agents on historical data.
        If force_train is True, ignores saved models and retrains.
        """
        print("\n[Orchestrator] Phase 1: Bootstrapping agents...")

        loaded_analysis = False
        loaded_diagnosis = False

        if not force_train:
            loaded_analysis = self.analysis_agent.load_model()
            loaded_diagnosis = self.diagnosis_agent.load_model()

        if not loaded_analysis:
            historical = self.sensor_agent.get_historical_data(hours=48)
            self.analysis_agent.train(historical)
        else:
            print("[Orchestrator] Skipped AnalysisAgent training (loaded from disk).")

        if not loaded_diagnosis:
            self.diagnosis_agent.train(labels_csv_path=self.labels_csv_path)
        else:
            print("[Orchestrator] Skipped DiagnosisAgent training (loaded from disk).")

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
        machines = self.sensor_agent.MACHINES
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


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predictive Maintenance — Agentic AI")
    parser.add_argument("--sensor-csv", type=str, default=None,
                        help="Path to sensor readings CSV file (real data mode)")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="Path to labeled fault data CSV file")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of monitoring cycles to run (default: 3)")
    parser.add_argument("--steps", type=int, default=2,
                        help="Timesteps per machine per cycle (default: 2)")
    parser.add_argument("--force-train", action="store_true",
                        help="Force retrain models even if .pkl files exist")
    args = parser.parse_args()

    orchestrator = OrchestratorAgent(
        sensor_csv_path=args.sensor_csv,
        labels_csv_path=args.labels_csv,
    )
    orchestrator.bootstrap(force_train=args.force_train)

    for cycle in range(args.cycles):
        print(f"\n--- Cycle {cycle + 1} ---")
        result = orchestrator.run_cycle(n_steps=args.steps)

    print("\n=== MAINTENANCE QUEUE ===")
    for task in orchestrator.maintenance_agent.get_prioritized_queue():
        print(f"  [{task.priority}] {task.machine_id} | {task.task_type} | "
              f"Scheduled: {task.scheduled_for.strftime('%Y-%m-%d %H:%M')}")

    print("\n=== SYSTEM SUMMARY ===")
    for mid, info in orchestrator.get_system_summary().items():
        print(f"  {mid}: {info['latest_fault']} | Severity: {info['severity']} | RUL: {info['estimated_rul']}h")
