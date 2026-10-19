"""Round Robin task scheduler baseline."""

from src.baselines.base import BaseScheduler
from src.simulator.environment import CloudSchedulingEnv


class RoundRobinScheduler(BaseScheduler):
    """Assigns incoming tasks cyclically across virtual machines in a fixed circular order."""

    def __init__(self):
        super().__init__(name="Round Robin")
        self.next_vm_idx = 0

    def select_action(self, env: CloudSchedulingEnv) -> int:
        mask = env.get_action_mask()
        num_vms = env.num_vms

        # Search for the next available valid VM in circular order
        for _ in range(num_vms):
            candidate = self.next_vm_idx
            self.next_vm_idx = (self.next_vm_idx + 1) % num_vms
            if mask[candidate]:
                return candidate

        return 0  # Fallback
