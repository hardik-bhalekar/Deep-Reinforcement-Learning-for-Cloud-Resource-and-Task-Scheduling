"""Unit tests for MetricsCollector."""

import unittest
from src.simulator.task import Task
from src.simulator.vm import VirtualMachine
from src.simulator.energy_model import EnergyModel
from src.simulator.metrics import MetricsCollector


class TestMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.vm1 = VirtualMachine(vm_id=0, mips=1000.0, ram_mb=4096.0,
                                  energy_model=EnergyModel(idle_power_w=100.0, max_power_w=200.0),
                                  cost_per_hour=0.36)
        self.vm2 = VirtualMachine(vm_id=1, mips=2000.0, ram_mb=4096.0,
                                  energy_model=EnergyModel(idle_power_w=100.0, max_power_w=200.0),
                                  cost_per_hour=0.36)

    def test_metrics_calculation(self):
        t1 = Task(task_id=0, length_mips=1000.0, ram_mb=1024.0, arrival_time=0.0, deadline=1.5)
        t2 = Task(task_id=1, length_mips=2000.0, ram_mb=1024.0, arrival_time=0.0, deadline=0.8)

        # Schedule t1 on vm1 (exec 1.0s, finishes at 1.0s <= deadline 1.5 -> Met)
        self.vm1.schedule_task(t1)
        # Schedule t2 on vm2 (exec 1.0s, finishes at 1.0s > deadline 0.8 -> Violated)
        self.vm2.schedule_task(t2)

        metrics = MetricsCollector.calculate_metrics([t1, t2], [self.vm1, self.vm2], scheduler_name="TestScheduler")

        self.assertEqual(metrics.scheduler_name, "TestScheduler")
        self.assertEqual(metrics.num_tasks, 2)
        self.assertEqual(metrics.num_vms, 2)
        self.assertEqual(metrics.makespan, 1.0)
        self.assertEqual(metrics.sla_violations_count, 1)
        self.assertEqual(metrics.sla_violation_rate, 0.5)
        self.assertEqual(metrics.task_completion_rate, 1.0)
        self.assertGreater(metrics.total_energy_joules, 0.0)
        self.assertGreater(metrics.total_cost, 0.0)

        # Verify to_dict output
        d = metrics.to_dict()
        self.assertIn("makespan", d)
        self.assertIn("total_energy_kwh", d)
        self.assertIn("avg_vm_utilization", d)

    def test_empty_tasks_raise(self):
        with self.assertRaises(ValueError):
            MetricsCollector.calculate_metrics([], [self.vm1])


if __name__ == "__main__":
    unittest.main()
