"""Unit tests for CloudSchedulingEnv."""

import unittest
import numpy as np
from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.simulator.task import Task
from src.simulator.vm import VirtualMachine


class TestCloudSchedulingEnv(unittest.TestCase):
    def setUp(self):
        self.workload = WorkloadGenerator(num_tasks=5, seed=42)
        self.env = CloudSchedulingEnv(workload_generator=self.workload)

    def test_initial_state_dimensions(self):
        obs, info = self.env.reset()
        self.assertEqual(len(obs), self.env.state_dim)
        # 4 task features + (4 per VM * 4 VMs) = 20
        self.assertEqual(self.env.state_dim, 20)
        self.assertEqual(self.env.action_dim, 4)
        self.assertIn("action_mask", info)
        self.assertEqual(len(info["action_mask"]), 4)

    def test_step_progression(self):
        obs, info = self.env.reset()
        self.assertEqual(self.env.current_task_idx, 0)

        # Step 0
        next_obs, reward, terminated, truncated, info = self.env.step(action=0)
        self.assertEqual(self.env.current_task_idx, 1)
        self.assertFalse(terminated)
        self.assertIsInstance(reward, float)
        self.assertEqual(len(next_obs), self.env.state_dim)

    def test_full_episode_termination_and_metrics(self):
        obs, info = self.env.reset()
        done = False
        step_count = 0

        while not done:
            action = step_count % self.env.num_vms
            obs, reward, terminated, truncated, info = self.env.step(action=action)
            done = terminated or truncated
            step_count += 1

        self.assertEqual(step_count, 5)
        self.assertTrue(done)
        self.assertIn("metrics", info)
        self.assertGreater(info["metrics"]["makespan"], 0.0)

    def test_action_masking_enforcement(self):
        # Create a VM with small RAM (512MB) and one with large RAM (8192MB)
        small_vm = VirtualMachine(vm_id=0, mips=1000.0, ram_mb=512.0)
        large_vm = VirtualMachine(vm_id=1, mips=1000.0, ram_mb=8192.0)
        # Task requires 2048MB RAM
        big_task = Task(task_id=0, length_mips=1000.0, ram_mb=2048.0)

        custom_env = CloudSchedulingEnv(vms=[small_vm, large_vm])
        custom_env.reset(tasks=[big_task])

        mask = custom_env.get_action_mask()
        self.assertFalse(mask[0])  # small VM cannot accommodate
        self.assertTrue(mask[1])   # large VM can accommodate


if __name__ == "__main__":
    unittest.main()
