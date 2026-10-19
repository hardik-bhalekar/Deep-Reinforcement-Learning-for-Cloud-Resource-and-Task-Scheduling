"""Unit tests for WorkloadGenerator."""

import unittest
from src.simulator.workload import WorkloadGenerator


class TestWorkloadGenerator(unittest.TestCase):
    def test_reproducibility_with_seed(self):
        gen1 = WorkloadGenerator(num_tasks=20, seed=123)
        gen2 = WorkloadGenerator(num_tasks=20, seed=123)

        tasks1 = gen1.generate()
        tasks2 = gen2.generate()

        self.assertEqual(len(tasks1), len(tasks2))
        for t1, t2 in zip(tasks1, tasks2):
            self.assertEqual(t1.task_id, t2.task_id)
            self.assertAlmostEqual(t1.length_mips, t2.length_mips)
            self.assertEqual(t1.ram_mb, t2.ram_mb)
            self.assertAlmostEqual(t1.arrival_time, t2.arrival_time)
            self.assertAlmostEqual(t1.deadline, t2.deadline)

    def test_different_seeds_produce_different_workloads(self):
        gen = WorkloadGenerator(num_tasks=10)
        tasks_a = gen.generate(seed=1)
        tasks_b = gen.generate(seed=2)

        lengths_a = [t.length_mips for t in tasks_a]
        lengths_b = [t.length_mips for t in tasks_b]
        self.assertNotEqual(lengths_a, lengths_b)

    def test_workload_patterns(self):
        for pattern in ["uniform", "heterogeneous", "bursty", "poisson_arrival"]:
            gen = WorkloadGenerator(num_tasks=15, workload_type=pattern, seed=42)
            tasks = gen.generate()
            self.assertEqual(len(tasks), 15)
            # Verify basic invariants
            for t in tasks:
                self.assertGreaterEqual(t.length_mips, gen.length_min * 0.5)
                self.assertGreaterEqual(t.deadline, t.arrival_time)
                self.assertIn(t.ram_mb, [512.0, 1024.0, 2048.0, 4096.0])


if __name__ == "__main__":
    unittest.main()
