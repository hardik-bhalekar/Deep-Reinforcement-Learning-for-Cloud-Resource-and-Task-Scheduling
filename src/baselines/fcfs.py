"""First Come, First Served (FCFS) task scheduler baseline."""

from src.baselines.base import BaseScheduler
from src.simulator.environment import CloudSchedulingEnv


class FCFSScheduler(BaseScheduler):
    """Assigns tasks in order of arrival to the VM with the earliest availability time."""

    def __init__(self):
        super().__init__(name="FCFS")

    def select_action(self, env: CloudSchedulingEnv) -> int:
        mask = env.get_action_mask()
        best_vm = 0
        earliest_available = float("inf")

        for i, vm in enumerate(env.vms):
            if mask[i] and vm.available_time < earliest_available:
                earliest_available = vm.available_time
                best_vm = i

        return best_vm
