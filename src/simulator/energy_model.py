"""Energy consumption model for virtual machines in cloud computing."""

from dataclasses import dataclass


@dataclass
class EnergyModel:
    """Calculates power and energy consumption based on the standard SPECpower model.

    Power consumption at CPU utilization u in [0, 1]:
        P(u) = P_idle + (P_max - P_idle) * u

    Attributes:
        idle_power_w: Power consumed by the VM/host when idle (Watts).
        max_power_w: Power consumed by the VM/host at 100% utilization (Watts).
    """

    idle_power_w: float = 100.0
    max_power_w: float = 250.0

    def __post_init__(self):
        if self.idle_power_w < 0:
            raise ValueError(f"idle_power_w must be non-negative, got {self.idle_power_w}")
        if self.max_power_w < self.idle_power_w:
            raise ValueError(
                f"max_power_w ({self.max_power_w}) cannot be less than idle_power_w ({self.idle_power_w})"
            )

    def power_at_utilization(self, utilization: float) -> float:
        """Calculate instantaneous power in Watts for a given CPU utilization in [0.0, 1.0]."""
        u = max(0.0, min(1.0, float(utilization)))
        return self.idle_power_w + (self.max_power_w - self.idle_power_w) * u

    def calculate_energy_joules(self, busy_time: float, total_span: float) -> float:
        """Calculate total energy in Joules over total_span where VM was busy for busy_time.

        Args:
            busy_time: Total seconds the VM was actively executing tasks.
            total_span: Total observation window or makespan (seconds).

        Returns:
            Total energy consumed in Joules (Watt-seconds).
        """
        busy_time = max(0.0, float(busy_time))
        total_span = max(busy_time, float(total_span))
        idle_time = max(0.0, total_span - busy_time)

        energy_busy = self.max_power_w * busy_time
        energy_idle = self.idle_power_w * idle_time
        return energy_busy + energy_idle

    @staticmethod
    def joules_to_kwh(joules: float) -> float:
        """Convert Joules to Kilowatt-hours (1 kWh = 3,600,000 Joules)."""
        return joules / 3.6e6
