"""Base interface for cloud task schedulers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from src.simulator.environment import CloudSchedulingEnv
from src.simulator.metrics import SimulationMetrics, MetricsCollector


class BaseScheduler(ABC):
    """Abstract base class for all cloud task scheduling algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, env: CloudSchedulingEnv) -> int:
        """Select a VM index for the current task in the environment.

        Args:
            env: Current cloud scheduling environment.

        Returns:
            Selected VM index (0 to num_vms - 1).
        """
        pass

    def run_episode(self, env: CloudSchedulingEnv, tasks: List = None, seed: int = None) -> SimulationMetrics:
        """Execute a full scheduling episode using this algorithm on the environment."""
        obs, info = env.reset(tasks=tasks, seed=seed)
        done = False

        while not done:
            action = self.select_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Return computed metrics
        metrics = MetricsCollector.calculate_metrics(env.tasks, env.vms, scheduler_name=self.name)
        return metrics
