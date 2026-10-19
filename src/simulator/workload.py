"""Reproducible workload generator for cloud tasks."""

from typing import List, Optional
import numpy as np
from src.simulator.task import Task


class WorkloadGenerator:
    """Generates synthetic task workloads for cloud scheduling experiments.

    Supports uniform, poisson arrival, heterogeneous (bimodal), and bursty patterns.
    All workloads are fully reproducible given a fixed random seed.
    """

    def __init__(
        self,
        num_tasks: int = 50,
        workload_type: str = "heterogeneous",
        seed: Optional[int] = 42,
        length_min: float = 1000.0,
        length_max: float = 10000.0,
        ram_min: float = 512.0,
        ram_max: float = 4096.0,
        arrival_rate: float = 1.0,
        deadline_slack: float = 1.8,
        ref_mips: float = 2000.0,
    ):
        self.num_tasks = num_tasks
        self.workload_type = workload_type
        self.seed = seed
        self.length_min = length_min
        self.length_max = length_max
        self.ram_min = ram_min
        self.ram_max = ram_max
        self.arrival_rate = arrival_rate
        self.deadline_slack = deadline_slack
        self.ref_mips = ref_mips

    def generate(self, seed: Optional[int] = None) -> List[Task]:
        """Generate a list of Task instances according to the configured distribution."""
        actual_seed = self.seed if seed is None else seed
        rng = np.random.RandomState(actual_seed)

        tasks: List[Task] = []
        current_arrival = 0.0

        for task_id in range(self.num_tasks):
            # 1. Determine task length (MIPS) based on pattern
            if self.workload_type == "uniform":
                length = rng.uniform(self.length_min, self.length_max)
                inter_arrival = rng.exponential(1.0 / self.arrival_rate) if self.arrival_rate > 0 else 0.5
            elif self.workload_type == "heterogeneous":
                # 70% light/interactive tasks, 30% compute-heavy batch tasks
                if rng.rand() < 0.70:
                    length = rng.uniform(self.length_min, self.length_min + 0.3 * (self.length_max - self.length_min))
                else:
                    length = rng.uniform(self.length_min + 0.6 * (self.length_max - self.length_min), self.length_max)
                inter_arrival = rng.exponential(1.0 / self.arrival_rate) if self.arrival_rate > 0 else 0.5
            elif self.workload_type == "bursty":
                # Bursts of tasks arrive at almost the same timestamp
                length = rng.uniform(self.length_min, self.length_max)
                if rng.rand() < 0.4:
                    inter_arrival = rng.uniform(0.01, 0.1)  # Burst arrival
                else:
                    inter_arrival = rng.uniform(1.5, 4.0)  # Idle interval
            elif self.workload_type == "poisson_arrival":
                length = rng.lognormal(mean=np.log(3000), sigma=0.5)
                length = np.clip(length, self.length_min, self.length_max)
                inter_arrival = rng.exponential(1.0 / self.arrival_rate) if self.arrival_rate > 0 else 0.5
            else:
                length = rng.uniform(self.length_min, self.length_max)
                inter_arrival = 0.5

            if task_id == 0:
                current_arrival = 0.0
            else:
                current_arrival += float(inter_arrival)

            # 2. Determine RAM requirement
            ram = float(rng.choice([512.0, 1024.0, 2048.0, 4096.0]))

            # 3. Compute realistic SLA deadline based on reference execution time
            est_duration = length / self.ref_mips
            deadline = current_arrival + (self.deadline_slack * est_duration)

            # 4. Priority (1 to 3)
            priority = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))

            tasks.append(
                Task(
                    task_id=task_id,
                    length_mips=round(float(length), 2),
                    ram_mb=ram,
                    arrival_time=round(float(current_arrival), 3),
                    deadline=round(float(deadline), 3),
                    priority=priority,
                )
            )

        return tasks
