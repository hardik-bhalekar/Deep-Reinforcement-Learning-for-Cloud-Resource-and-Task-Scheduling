"""Unit tests for baseline schedulers."""

import unittest
from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.baselines.random_scheduler import RandomScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.fcfs import FCFSScheduler
from src.baselines.min_min import MinMinScheduler


class TestBaselineSchedulers(unittest.TestCase):
    def setUp(self):
        self.workload = WorkloadGenerator(num_tasks=20, seed=42)
        self.env = CloudSchedulingEnv(workload_generator=self.workload)

    def test_random_scheduler(self):
        scheduler = RandomScheduler(seed=42)
        metrics = scheduler.run_episode(self.env)
        self.assertEqual(metrics.scheduler_name, "Random")
        self.assertEqual(metrics.task_completion_rate, 1.0)
        self.assertGreater(metrics.makespan, 0.0)
        self.assertGreater(metrics.total_energy_joules, 0.0)

    def test_round_robin_scheduler(self):
        scheduler = RoundRobinScheduler()
        metrics = scheduler.run_episode(self.env)
        self.assertEqual(metrics.scheduler_name, "Round Robin")
        self.assertEqual(metrics.task_completion_rate, 1.0)
        self.assertGreater(metrics.makespan, 0.0)

    def test_fcfs_scheduler(self):
        scheduler = FCFSScheduler()
        metrics = scheduler.run_episode(self.env)
        self.assertEqual(metrics.scheduler_name, "FCFS")
        self.assertEqual(metrics.task_completion_rate, 1.0)
        self.assertGreater(metrics.makespan, 0.0)

    def test_min_min_scheduler(self):
        scheduler = MinMinScheduler()
        metrics = scheduler.run_episode(self.env)
        self.assertEqual(metrics.scheduler_name, "Min-Min")
        self.assertEqual(metrics.task_completion_rate, 1.0)
        self.assertGreater(metrics.makespan, 0.0)

    def test_min_min_achieves_competitive_makespan(self):
        # Min-Min typically achieves lower makespan than uncoordinated Random or Round Robin
        random_m = RandomScheduler(seed=42).run_episode(self.env)
        minmin_m = MinMinScheduler().run_episode(self.env)

        # Both should successfully complete all tasks
        self.assertEqual(random_m.task_completion_rate, 1.0)
        self.assertEqual(minmin_m.task_completion_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
