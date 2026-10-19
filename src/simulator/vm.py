"""Virtual Machine model for cloud resource management."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.simulator.task import Task
from src.simulator.energy_model import EnergyModel


@dataclass
class VirtualMachine:
    """Represents a Virtual Machine (VM) host in a cloud data center.

    Attributes:
        vm_id: Unique identifier for the VM.
        mips: Processing capacity in Million Instructions Per Second (MIPS).
        ram_mb: Total RAM capacity in Megabytes.
        energy_model: Power and energy consumption model.
        cost_per_hour: Financial cost of running this VM per hour ($/hr).
    """

    vm_id: int
    mips: float
    ram_mb: float
    energy_model: EnergyModel = field(default_factory=EnergyModel)
    cost_per_hour: float = 0.10

    # Runtime scheduling state
    available_time: float = 0.0
    total_busy_time: float = 0.0
    assigned_tasks: List[Task] = field(default_factory=list)

    def can_accommodate(self, task: Task) -> bool:
        """Check if the VM meets the resource requirements of the task."""
        return task.ram_mb <= self.ram_mb

    def estimate_execution_time(self, task: Task) -> float:
        """Calculate the execution duration of a task on this VM in seconds."""
        if self.mips <= 0:
            raise ValueError(f"VM {self.vm_id} has invalid MIPS: {self.mips}")
        return float(task.length_mips) / float(self.mips)

    def estimate_completion_time(self, task: Task, current_time: float = 0.0) -> float:
        """Estimate the timestamp at which this task would finish if assigned to this VM."""
        start_time = max(self.available_time, task.arrival_time, current_time)
        return start_time + self.estimate_execution_time(task)

    def schedule_task(self, task: Task, current_time: float = 0.0) -> None:
        """Assign and commit a task to this VM's execution timeline."""
        if not self.can_accommodate(task):
            raise ValueError(
                f"VM {self.vm_id} (RAM: {self.ram_mb}MB) cannot accommodate Task {task.task_id} (RAM: {task.ram_mb}MB)"
            )
        start_time = max(self.available_time, task.arrival_time, current_time)
        exec_time = self.estimate_execution_time(task)
        task.assign(self.vm_id, start_time, exec_time)
        self.available_time = task.finish_time
        self.total_busy_time += exec_time
        self.assigned_tasks.append(task)

    def get_utilization(self, total_time: float) -> float:
        """Calculate CPU utilization over the total observation duration."""
        if total_time <= 0:
            return 0.0
        return min(1.0, self.total_busy_time / total_time)

    def calculate_energy(self, total_time: float) -> float:
        """Calculate total energy consumed by this VM in Joules."""
        return self.energy_model.calculate_energy_joules(self.total_busy_time, total_time)

    def calculate_cost(self, total_time: float) -> float:
        """Calculate total operational cost in US dollars."""
        hours = max(0.0, total_time) / 3600.0
        return hours * self.cost_per_hour

    def reset(self) -> None:
        """Reset the VM's execution queue and counters for a new simulation episode."""
        self.available_time = 0.0
        self.total_busy_time = 0.0
        self.assigned_tasks.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert VM characteristics and performance to a dictionary."""
        return {
            "vm_id": self.vm_id,
            "mips": self.mips,
            "ram_mb": self.ram_mb,
            "idle_power_w": self.energy_model.idle_power_w,
            "max_power_w": self.energy_model.max_power_w,
            "cost_per_hour": self.cost_per_hour,
            "available_time": self.available_time,
            "total_busy_time": self.total_busy_time,
            "num_assigned_tasks": len(self.assigned_tasks),
        }
