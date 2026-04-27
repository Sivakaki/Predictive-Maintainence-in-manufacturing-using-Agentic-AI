"""
Predictive Maintenance — Maintenance Agent
maintenance_agent.py — Schedules maintenance tasks based on diagnosis
"""

from datetime import timedelta
from typing import Optional

from models import DiagnosisReport, MaintenanceTask
from inventory_agent import InventoryAgent

class MaintenanceAgent:
    """
    Plans and schedules maintenance tasks based on diagnosis.
    Applies priority-based scheduling logic.
    """

    def __init__(self, inventory_agent: Optional[InventoryAgent] = None):
        self.task_queue: list[MaintenanceTask] = []
        self.completed_tasks: list[MaintenanceTask] = []
        self.inventory_agent = inventory_agent
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

        # Check inventory if we have an inventory agent
        inv_note = ""
        missing_parts = False
        if self.inventory_agent:
            parts_avail, msg = self.inventory_agent.check_and_reserve(diagnosis.fault_type)
            inv_note = f"| Inv: {msg} "
            if not parts_avail:
                missing_parts = True

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
                f"Action: {diagnosis.recommended_action} {inv_note}"
            ),
            estimated_downtime_hours=downtime[diagnosis.severity],
        )
        self.task_queue.append(task)
        return task

    def get_prioritized_queue(self) -> list[MaintenanceTask]:
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        return sorted(self.task_queue, key=lambda t: (priority_order[t.priority], t.scheduled_for))
