"""Random task scheduler baseline."""

from typing import Optional
import numpy as np
from src.baselines.base import BaseScheduler
from src.simulator.environment import CloudSchedulingEnv


class RandomScheduler(BaseScheduler):
    """Assigns each arriving task to a uniformly chosen valid virtual machine."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="Random")
        self.rng = np.random.RandomState(seed)

    def select_action(self, env: CloudSchedulingEnv) -> int:
        mask = env.get_action_mask()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0
        return int(self.rng.choice(valid_indices))
