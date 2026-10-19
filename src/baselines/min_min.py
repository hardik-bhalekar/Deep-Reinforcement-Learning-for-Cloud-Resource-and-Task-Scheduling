"""Min-Min heuristic task scheduler baseline."""

from src.baselines.base import BaseScheduler
from src.simulator.environment import CloudSchedulingEnv


class MinMinScheduler(BaseScheduler):
    """Assigns each task to the virtual machine that yields the minimum completion time."""

    def __init__(self):
        super().__init__(name="Min-Min")

    def select_action(self, env: CloudSchedulingEnv) -> int:
        mask = env.get_action_mask()
        task = env.tasks[env.current_task_idx]
        current_time = task.arrival_time

        best_vm = 0
        min_completion_time = float("inf")

        for i, vm in enumerate(env.vms):
            if mask[i]:
                est_completion = vm.estimate_completion_time(task, current_time)
                if est_completion < min_completion_time:
                    min_completion_time = est_completion
                    best_vm = i

        return best_vm
