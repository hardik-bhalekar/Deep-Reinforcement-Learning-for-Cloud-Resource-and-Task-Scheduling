"""Cloud Scheduling Environment adhering to standard RL environment interfaces."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.simulator.task import Task
from src.simulator.vm import VirtualMachine
from src.simulator.energy_model import EnergyModel
from src.simulator.workload import WorkloadGenerator
from src.simulator.metrics import MetricsCollector, SimulationMetrics


class CloudSchedulingEnv:
    """Simulated cloud resource and task scheduling environment.

    State space:
        Fixed-size float32 array encoding:
        - Current task properties (normalized length, RAM requirement, deadline urgency, progress remaining)
        - Per-VM status (normalized MIPS, RAM capacity, queue delay / waiting time, estimated completion time)

    Action space:
        Discrete integer [0, num_vms - 1] representing the VM assigned to execute the current task.
    """

    def __init__(
        self,
        vms: Optional[List[VirtualMachine]] = None,
        workload_generator: Optional[WorkloadGenerator] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        max_horizon: float = 200.0,
    ):
        # 1. Initialize VMs (default: 4 heterogeneous VMs)
        if vms is None:
            self.vms = [
                VirtualMachine(vm_id=0, mips=1000.0, ram_mb=4096.0, cost_per_hour=0.05,
                               energy_model=EnergyModel(idle_power_w=80.0, max_power_w=200.0)),
                VirtualMachine(vm_id=1, mips=2000.0, ram_mb=8192.0, cost_per_hour=0.10,
                               energy_model=EnergyModel(idle_power_w=110.0, max_power_w=250.0)),
                VirtualMachine(vm_id=2, mips=3000.0, ram_mb=8192.0, cost_per_hour=0.15,
                               energy_model=EnergyModel(idle_power_w=140.0, max_power_w=300.0)),
                VirtualMachine(vm_id=3, mips=4000.0, ram_mb=16384.0, cost_per_hour=0.25,
                               energy_model=EnergyModel(idle_power_w=180.0, max_power_w=400.0)),
            ]
        else:
            self.vms = vms

        self.num_vms = len(self.vms)

        # 2. Workload generator
        self.workload_generator = workload_generator or WorkloadGenerator()
        self.tasks: List[Task] = []
        self.current_task_idx: int = 0
        self.max_horizon = max_horizon

        # 3. Normalization reference anchors
        self.max_mips = max(vm.mips for vm in self.vms)
        self.max_ram = max(vm.ram_mb for vm in self.vms)
        self.max_task_length = self.workload_generator.length_max
        self.ref_energy = max(vm.energy_model.max_power_w for vm in self.vms) * 5.0  # reference ~5s execution

        # 4. Multi-objective reward weights
        self.reward_weights = reward_weights or {
            "w_makespan": 0.4,
            "w_energy": 0.3,
            "w_sla": 0.2,
            "w_cost": 0.1,
        }

        # 5. Observation and action space dimensions
        # 4 task features + (4 features per VM * num_vms)
        self.state_dim = 4 + (4 * self.num_vms)
        self.action_dim = self.num_vms

    def reset(self, tasks: Optional[List[Task]] = None, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state with fresh VM queues and task sequence."""
        for vm in self.vms:
            vm.reset()

        if tasks is not None:
            self.tasks = tasks
            for t in self.tasks:
                t.reset()
        else:
            self.tasks = self.workload_generator.generate(seed=seed)

        self.current_task_idx = 0
        observation = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "current_task_idx": self.current_task_idx,
            "total_tasks": len(self.tasks),
        }
        return observation, info

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask where mask[i] is True if VM i can accommodate current task."""
        mask = np.zeros(self.num_vms, dtype=bool)
        if self.current_task_idx >= len(self.tasks):
            mask[:] = True
            return mask

        current_task = self.tasks[self.current_task_idx]
        for i, vm in enumerate(self.vms):
            if vm.can_accommodate(current_task):
                mask[i] = True

        # Fallback: if no VM satisfies constraint, allow all VMs to prevent deadlocks
        if not np.any(mask):
            mask[:] = True
        return mask

    def _get_observation(self) -> np.ndarray:
        """Construct normalized fixed-size observation vector."""
        obs = np.zeros(self.state_dim, dtype=np.float32)
        total_tasks = max(1, len(self.tasks))

        if self.current_task_idx >= len(self.tasks):
            return obs

        task = self.tasks[self.current_task_idx]

        # 1. Current Task Features (4 dims)
        length_norm = min(1.0, task.length_mips / self.max_task_length)
        ram_norm = min(1.0, task.ram_mb / self.max_ram)
        remaining_tasks_ratio = (total_tasks - self.current_task_idx) / total_tasks
        urgency_norm = min(1.0, max(0.0, task.deadline / self.max_horizon))

        obs[0] = length_norm
        obs[1] = ram_norm
        obs[2] = remaining_tasks_ratio
        obs[3] = urgency_norm

        # 2. Per-VM Features (4 dims per VM)
        offset = 4
        current_time = task.arrival_time
        for i, vm in enumerate(self.vms):
            mips_norm = vm.mips / self.max_mips
            ram_capacity_norm = vm.ram_mb / self.max_ram
            queue_delay = max(0.0, vm.available_time - current_time)
            queue_delay_norm = min(1.0, queue_delay / self.max_horizon)
            est_completion = vm.estimate_completion_time(task, current_time)
            est_comp_norm = min(1.0, est_completion / self.max_horizon)

            obs[offset] = mips_norm
            obs[offset + 1] = ram_capacity_norm
            obs[offset + 2] = queue_delay_norm
            obs[offset + 3] = est_comp_norm
            offset += 4

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Schedule current task on the chosen VM and advance simulation state."""
        if self.current_task_idx >= len(self.tasks):
            raise IndexError("Cannot step when all tasks have already been scheduled.")

        # Ensure valid action range
        action = int(action)
        action_mask = self.get_action_mask()
        invalid_action_taken = not action_mask[action]

        task = self.tasks[self.current_task_idx]
        selected_vm = self.vms[action]

        # Schedule the task on the VM
        prev_available_time = selected_vm.available_time
        selected_vm.schedule_task(task, current_time=task.arrival_time)

        # Calculate reward components
        reward = self._calculate_reward(task, selected_vm, prev_available_time, invalid_action_taken)

        # Advance pointer
        self.current_task_idx += 1
        terminated = (self.current_task_idx >= len(self.tasks))
        truncated = False

        next_obs = self._get_observation()
        info: Dict[str, Any] = {
            "action_mask": self.get_action_mask(),
            "current_task_idx": self.current_task_idx,
            "total_tasks": len(self.tasks),
            "assigned_vm": action,
            "invalid_action": invalid_action_taken,
        }

        if terminated:
            # Collect full benchmark metrics
            metrics = MetricsCollector.calculate_metrics(self.tasks, self.vms, scheduler_name="DRL/Env")
            info["metrics"] = metrics.to_dict()

        return next_obs, float(reward), terminated, truncated, info

    def _calculate_reward(
        self,
        task: Task,
        vm: VirtualMachine,
        prev_available: float,
        invalid_action: bool,
    ) -> float:
        """Compute multi-objective normalized step reward.

        Reward components:
        1. Turnaround time: Penalizes total turnaround (waiting + execution).
        2. Energy: Penalizes incremental energy consumed by the task.
        3. SLA Deadline: Bonus for meeting SLA, penalty for deadline violation.
        4. Invalid action: Strong negative penalty for resource constraint violation.
        """
        if invalid_action:
            return -2.0  # Substantial penalty for violating constraint

        # 1. Turnaround time component (normalized against max horizon)
        r_time = -min(2.0, (task.turnaround_time or 0.0) / 10.0)

        # 2. Energy component: Energy consumed during task execution
        task_energy_joules = vm.energy_model.max_power_w * (task.execution_time or 0.0)
        r_energy = -min(2.0, task_energy_joules / self.ref_energy)

        # 3. SLA component
        r_sla = 0.5 if task.deadline_met else -1.0

        # 4. Cost component
        task_cost = (vm.cost_per_hour / 3600.0) * (task.execution_time or 0.0)
        r_cost = -min(2.0, task_cost * 1000.0)

        w = self.reward_weights
        total_reward = (
            w.get("w_makespan", 0.4) * r_time
            + w.get("w_energy", 0.3) * r_energy
            + w.get("w_sla", 0.2) * r_sla
            + w.get("w_cost", 0.1) * r_cost
        )
        return float(total_reward)
