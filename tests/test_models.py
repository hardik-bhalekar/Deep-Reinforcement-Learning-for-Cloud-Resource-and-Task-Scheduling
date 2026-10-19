"""Unit tests for Task, VirtualMachine, and EnergyModel."""

import unittest
from src.simulator.task import Task
from src.simulator.vm import VirtualMachine
from src.simulator.energy_model import EnergyModel


class TestTaskModel(unittest.TestCase):
    def test_task_creation_and_attributes(self):
        task = Task(task_id=1, length_mips=2000.0, ram_mb=1024.0, arrival_time=1.0, deadline=5.0, priority=2)
        self.assertEqual(task.task_id, 1)
        self.assertEqual(task.length_mips, 2000.0)
        self.assertEqual(task.ram_mb, 1024.0)
        self.assertEqual(task.arrival_time, 1.0)
        self.assertEqual(task.deadline, 5.0)
        self.assertEqual(task.priority, 2)
        self.assertIsNone(task.assigned_vm_id)

    def test_task_assignment_and_sla(self):
        task = Task(task_id=2, length_mips=1000.0, ram_mb=512.0, arrival_time=0.5, deadline=2.0)
        # Assign to VM 0, starting at time 1.0, duration 0.5s -> finish at 1.5s (within deadline 2.0)
        task.assign(vm_id=0, start_time=1.0, execution_time=0.5)

        self.assertEqual(task.assigned_vm_id, 0)
        self.assertEqual(task.start_time, 1.0)
        self.assertEqual(task.execution_time, 0.5)
        self.assertEqual(task.finish_time, 1.5)
        self.assertEqual(task.waiting_time, 0.5)  # 1.0 - 0.5
        self.assertEqual(task.turnaround_time, 1.0)  # 1.5 - 0.5
        self.assertTrue(task.deadline_met)

    def test_task_sla_violation(self):
        task = Task(task_id=3, length_mips=5000.0, ram_mb=512.0, arrival_time=0.0, deadline=2.0)
        # Execution finishes at 3.0s -> deadline violated
        task.assign(vm_id=1, start_time=0.0, execution_time=3.0)
        self.assertFalse(task.deadline_met)

    def test_task_reset(self):
        task = Task(task_id=4, length_mips=1000.0, ram_mb=512.0)
        task.assign(vm_id=0, start_time=0.0, execution_time=1.0)
        task.reset()
        self.assertIsNone(task.assigned_vm_id)
        self.assertIsNone(task.finish_time)


class TestEnergyModel(unittest.TestCase):
    def test_power_at_utilization(self):
        model = EnergyModel(idle_power_w=100.0, max_power_w=200.0)
        self.assertEqual(model.power_at_utilization(0.0), 100.0)
        self.assertEqual(model.power_at_utilization(1.0), 200.0)
        self.assertEqual(model.power_at_utilization(0.5), 150.0)

    def test_calculate_energy_joules(self):
        model = EnergyModel(idle_power_w=100.0, max_power_w=200.0)
        # Busy 10s at 200W = 2000J. Idle 10s at 100W = 1000J. Total span 20s -> 3000 Joules
        energy = model.calculate_energy_joules(busy_time=10.0, total_span=20.0)
        self.assertEqual(energy, 3000.0)

    def test_joules_to_kwh(self):
        kwh = EnergyModel.joules_to_kwh(3.6e6)
        self.assertAlmostEqual(kwh, 1.0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            EnergyModel(idle_power_w=-10.0)
        with self.assertRaises(ValueError):
            EnergyModel(idle_power_w=300.0, max_power_w=200.0)


class TestVirtualMachine(unittest.TestCase):
    def setUp(self):
        self.vm = VirtualMachine(
            vm_id=0,
            mips=1000.0,
            ram_mb=4096.0,
            energy_model=EnergyModel(idle_power_w=100.0, max_power_w=200.0),
            cost_per_hour=0.36,  # $0.0001 per second
        )

    def test_can_accommodate(self):
        fit_task = Task(task_id=1, length_mips=1000.0, ram_mb=2048.0)
        unfit_task = Task(task_id=2, length_mips=1000.0, ram_mb=8192.0)
        self.assertTrue(self.vm.can_accommodate(fit_task))
        self.assertFalse(self.vm.can_accommodate(unfit_task))

    def test_execution_time_estimation(self):
        task = Task(task_id=1, length_mips=2500.0, ram_mb=1024.0)
        exec_time = self.vm.estimate_execution_time(task)
        self.assertEqual(exec_time, 2.5)  # 2500 / 1000

    def test_schedule_task_sequential(self):
        t1 = Task(task_id=1, length_mips=1000.0, ram_mb=1024.0, arrival_time=0.0)
        t2 = Task(task_id=2, length_mips=2000.0, ram_mb=1024.0, arrival_time=0.5)

        self.vm.schedule_task(t1)
        self.assertEqual(t1.start_time, 0.0)
        self.assertEqual(t1.finish_time, 1.0)
        self.assertEqual(self.vm.available_time, 1.0)

        self.vm.schedule_task(t2)
        # t2 arrived at 0.5s, but VM busy until 1.0s -> starts at 1.0s, finishes at 3.0s
        self.assertEqual(t2.start_time, 1.0)
        self.assertEqual(t2.finish_time, 3.0)
        self.assertEqual(self.vm.available_time, 3.0)
        self.assertEqual(self.vm.total_busy_time, 3.0)

    def test_energy_and_cost_calculation(self):
        t1 = Task(task_id=1, length_mips=1000.0, ram_mb=1024.0)
        self.vm.schedule_task(t1)  # busy for 1.0s
        # Total span 10s: 1s busy at 200W (200J) + 9s idle at 100W (900J) = 1100J
        energy = self.vm.calculate_energy(10.0)
        self.assertEqual(energy, 1100.0)
        cost = self.vm.calculate_cost(3600.0)  # 1 hour
        self.assertAlmostEqual(cost, 0.36)


if __name__ == "__main__":
    unittest.main()
