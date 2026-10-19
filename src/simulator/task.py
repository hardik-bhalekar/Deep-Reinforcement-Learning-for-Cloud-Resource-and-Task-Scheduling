"""Task data model for cloud scheduling."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Task:
    """Represents a computational task to be scheduled in the cloud.

    Attributes:
        task_id: Unique identifier for the task.
        length_mips: Workload size in Million Instructions (MI).
        ram_mb: RAM memory requirement in megabytes.
        arrival_time: Time at which the task arrives at the scheduler (seconds).
        deadline: SLA deadline by which the task should complete (seconds).
        priority: Priority of the task (higher value means higher priority).
    """

    task_id: int
    length_mips: float
    ram_mb: float
    arrival_time: float = 0.0
    deadline: float = float("inf")
    priority: int = 1

    # Runtime scheduling state
    assigned_vm_id: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    waiting_time: Optional[float] = None
    execution_time: Optional[float] = None
    turnaround_time: Optional[float] = None
    deadline_met: Optional[bool] = None

    def assign(self, vm_id: int, start_time: float, execution_time: float) -> None:
        """Record the scheduling assignment and execution timing for the task."""
        self.assigned_vm_id = vm_id
        self.start_time = max(start_time, self.arrival_time)
        self.execution_time = execution_time
        self.finish_time = self.start_time + execution_time
        self.waiting_time = self.start_time - self.arrival_time
        self.turnaround_time = self.finish_time - self.arrival_time
        self.deadline_met = self.finish_time <= self.deadline

    def reset(self) -> None:
        """Reset the scheduling state of the task for a fresh simulation run."""
        self.assigned_vm_id = None
        self.start_time = None
        self.finish_time = None
        self.waiting_time = None
        self.execution_time = None
        self.turnaround_time = None
        self.deadline_met = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task details and execution results to a dictionary."""
        return {
            "task_id": self.task_id,
            "length_mips": self.length_mips,
            "ram_mb": self.ram_mb,
            "arrival_time": self.arrival_time,
            "deadline": self.deadline,
            "priority": self.priority,
            "assigned_vm_id": self.assigned_vm_id,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "waiting_time": self.waiting_time,
            "execution_time": self.execution_time,
            "turnaround_time": self.turnaround_time,
            "deadline_met": self.deadline_met,
        }
