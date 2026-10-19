"""Metrics collector for cloud scheduling evaluation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from src.simulator.task import Task
from src.simulator.vm import VirtualMachine


@dataclass
class SimulationMetrics:
    """Summary of performance metrics for an evaluation run."""

    scheduler_name: str
    num_tasks: int
    num_vms: int
    makespan: float
    avg_turnaround_time: float
    avg_waiting_time: float
    total_energy_joules: float
    total_energy_kwh: float
    avg_vm_utilization: float
    sla_violation_rate: float
    sla_violations_count: int
    total_cost: float
    degree_of_imbalance: float
    task_completion_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "scheduler_name": self.scheduler_name,
            "num_tasks": self.num_tasks,
            "num_vms": self.num_vms,
            "makespan": round(self.makespan, 3),
            "avg_turnaround_time": round(self.avg_turnaround_time, 3),
            "avg_waiting_time": round(self.avg_waiting_time, 3),
            "total_energy_joules": round(self.total_energy_joules, 2),
            "total_energy_kwh": round(self.total_energy_kwh, 6),
            "avg_vm_utilization": round(self.avg_vm_utilization, 4),
            "sla_violation_rate": round(self.sla_violation_rate, 4),
            "sla_violations_count": self.sla_violations_count,
            "total_cost": round(self.total_cost, 4),
            "degree_of_imbalance": round(self.degree_of_imbalance, 4),
            "task_completion_rate": round(self.task_completion_rate, 4),
        }


class MetricsCollector:
    """Computes, aggregates, and exports scheduling performance metrics."""

    @staticmethod
    def calculate_metrics(
        tasks: List[Task],
        vms: List[VirtualMachine],
        scheduler_name: str = "Unknown",
    ) -> SimulationMetrics:
        """Compute comprehensive metrics from completed tasks and VM states."""
        if not tasks:
            raise ValueError("Cannot calculate metrics on an empty task list.")
        if not vms:
            raise ValueError("Cannot calculate metrics on an empty VM list.")

        # Filter completed tasks
        completed_tasks = [t for t in tasks if t.finish_time is not None]
        num_completed = len(completed_tasks)
        task_completion_rate = num_completed / len(tasks)

        if num_completed == 0:
            return SimulationMetrics(
                scheduler_name=scheduler_name,
                num_tasks=len(tasks),
                num_vms=len(vms),
                makespan=0.0,
                avg_turnaround_time=0.0,
                avg_waiting_time=0.0,
                total_energy_joules=0.0,
                total_energy_kwh=0.0,
                avg_vm_utilization=0.0,
                sla_violation_rate=0.0,
                sla_violations_count=0,
                total_cost=0.0,
                degree_of_imbalance=0.0,
                task_completion_rate=0.0,
            )

        # 1. Makespan: latest finish time across all tasks
        makespan = float(max(t.finish_time for t in completed_tasks))

        # 2. Time metrics
        avg_turnaround = float(np.mean([t.turnaround_time for t in completed_tasks]))
        avg_waiting = float(np.mean([t.waiting_time for t in completed_tasks]))

        # 3. SLA / Deadline violations
        sla_violations = sum(1 for t in completed_tasks if not t.deadline_met)
        sla_violation_rate = sla_violations / num_completed

        # 4. Energy and cost calculations per VM
        total_energy_joules = 0.0
        total_cost = 0.0
        utilizations = []
        busy_times = []

        for vm in vms:
            total_energy_joules += vm.calculate_energy(makespan)
            total_cost += vm.calculate_cost(makespan)
            utilizations.append(vm.get_utilization(makespan))
            busy_times.append(vm.total_busy_time)

        total_energy_kwh = total_energy_joules / 3.6e6
        avg_vm_utilization = float(np.mean(utilizations))

        # 5. Degree of Imbalance (DI): (T_max - T_min) / T_avg
        t_max = max(busy_times)
        t_min = min(busy_times)
        t_avg = np.mean(busy_times)
        degree_of_imbalance = float((t_max - t_min) / t_avg) if t_avg > 0 else 0.0

        return SimulationMetrics(
            scheduler_name=scheduler_name,
            num_tasks=len(tasks),
            num_vms=len(vms),
            makespan=makespan,
            avg_turnaround_time=avg_turnaround,
            avg_waiting_time=avg_waiting,
            total_energy_joules=total_energy_joules,
            total_energy_kwh=total_energy_kwh,
            avg_vm_utilization=avg_vm_utilization,
            sla_violation_rate=sla_violation_rate,
            sla_violations_count=sla_violations,
            total_cost=total_cost,
            degree_of_imbalance=degree_of_imbalance,
            task_completion_rate=task_completion_rate,
        )
