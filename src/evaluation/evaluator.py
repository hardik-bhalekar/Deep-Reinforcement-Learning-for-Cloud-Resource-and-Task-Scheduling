"""Evaluation pipeline comparing DRL with baseline scheduling algorithms."""

import json
import os
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.simulator.metrics import SimulationMetrics, MetricsCollector
from src.baselines.random_scheduler import RandomScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.fcfs import FCFSScheduler
from src.baselines.min_min import MinMinScheduler
from src.dqn.agent import DQNAgent


class BenchmarkEvaluator:
    """Runs fair comparative evaluations of DRL and baseline algorithms across multiple seeds."""

    def __init__(
        self,
        env: CloudSchedulingEnv,
        dqn_agent: Optional[DQNAgent] = None,
        test_seeds: Optional[List[int]] = None,
        results_dir: str = "results",
    ):
        self.env = env
        self.dqn_agent = dqn_agent
        self.test_seeds = test_seeds or [101, 102, 103, 104, 105]
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.baselines = {
            "Random": RandomScheduler(),
            "Round Robin": RoundRobinScheduler(),
            "FCFS": FCFSScheduler(),
            "Min-Min": MinMinScheduler(),
        }

    def _run_single_scheduler(self, name: str, tasks: List) -> SimulationMetrics:
        """Execute a scheduler on a fixed task list and return computed metrics."""
        obs, info = self.env.reset(tasks=tasks)
        done = False

        if name == "DQN":
            if self.dqn_agent is None:
                raise ValueError("Cannot evaluate DQN without a loaded DQNAgent.")
            while not done:
                action = self.dqn_agent.select_action(obs, info["action_mask"], evaluate=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
        else:
            scheduler = self.baselines[name]
            while not done:
                action = scheduler.select_action(self.env)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

        return MetricsCollector.calculate_metrics(self.env.tasks, self.env.vms, scheduler_name=name)

    def run_benchmark(self, num_tasks: int = 50, workload_type: str = "heterogeneous") -> Dict[str, Any]:
        """Run all algorithms on identical generated workloads across test seeds."""
        algorithms = list(self.baselines.keys())
        if self.dqn_agent is not None:
            algorithms.append("DQN")

        raw_results: Dict[str, List[Dict[str, Any]]] = {algo: [] for algo in algorithms}

        for seed in self.test_seeds:
            # Generate identical evaluation workload for this seed
            workload_gen = WorkloadGenerator(
                num_tasks=num_tasks,
                workload_type=workload_type,
                seed=seed,
            )
            base_tasks = workload_gen.generate()

            for algo in algorithms:
                # Deep copy tasks for isolated evaluation run
                tasks_copy = workload_gen.generate()
                metrics = self._run_single_scheduler(algo, tasks_copy)
                raw_results[algo].append(metrics.to_dict())

        # Compute summary statistics (mean & std)
        summary: Dict[str, Dict[str, Any]] = {}
        metric_keys = [
            "makespan",
            "avg_turnaround_time",
            "avg_waiting_time",
            "total_energy_kwh",
            "avg_vm_utilization",
            "sla_violation_rate",
            "total_cost",
            "degree_of_imbalance",
        ]

        for algo in algorithms:
            summary[algo] = {}
            for k in metric_keys:
                vals = [run[k] for run in raw_results[algo]]
                summary[algo][f"{k}_mean"] = float(np.mean(vals))
                summary[algo][f"{k}_std"] = float(np.std(vals))

        # Save results to JSON and CSV
        results_payload = {
            "test_seeds": self.test_seeds,
            "num_tasks": num_tasks,
            "workload_type": workload_type,
            "summary": summary,
            "raw_results": raw_results,
        }

        json_path = os.path.join(self.results_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(results_payload, f, indent=2)

        # Build DataFrame table
        table_rows = []
        for algo in algorithms:
            row = {"Algorithm": algo}
            for k in metric_keys:
                mean_val = summary[algo][f"{k}_mean"]
                std_val = summary[algo][f"{k}_std"]
                row[k] = f"{mean_val:.3f} ± {std_val:.3f}"
            table_rows.append(row)
        df_summary = pd.DataFrame(table_rows)
        csv_path = os.path.join(self.results_dir, "benchmark_summary.csv")
        df_summary.to_csv(csv_path, index=False)

        return results_payload

    def generate_comparison_plots(self, benchmark_data: Dict[str, Any]) -> List[str]:
        """Generate and save publication-quality bar charts comparing all schedulers."""
        summary = benchmark_data["summary"]
        algorithms = list(summary.keys())
        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#7ED321", "#9013FE"][:len(algorithms)]

        generated_plots = []

        # 1. Multi-metric 4-panel comparison
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        metrics_to_plot = [
            ("makespan", "Makespan (seconds - lower is better)", axes[0, 0]),
            ("total_energy_kwh", "Total Energy (kWh - lower is better)", axes[0, 1]),
            ("sla_violation_rate", "SLA Violation Rate (lower is better)", axes[1, 0]),
            ("avg_vm_utilization", "Avg VM Utilization (higher is better)", axes[1, 1]),
        ]

        for metric_name, label, ax in metrics_to_plot:
            means = [summary[algo][f"{metric_name}_mean"] for algo in algorithms]
            stds = [summary[algo][f"{metric_name}_std"] for algo in algorithms]

            bars = ax.bar(algorithms, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_ylabel(metric_name)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.annotate(
                    f"{mean_val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        multi_path = os.path.join(self.plots_dir, "comprehensive_comparison.png")
        plt.savefig(multi_path, dpi=200)
        plt.close()
        generated_plots.append(multi_path)

        # 2. Individual plots
        for metric_name, title in [
            ("makespan", "Makespan by Algorithm"),
            ("total_energy_kwh", "Energy Consumption (kWh) by Algorithm"),
            ("sla_violation_rate", "SLA Deadline Violations by Algorithm"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            means = [summary[algo][f"{metric_name}_mean"] for algo in algorithms]
            stds = [summary[algo][f"{metric_name}_std"] for algo in algorithms]
            ax.bar(algorithms, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel(metric_name)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, f"{metric_name}_comparison.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            generated_plots.append(plot_path)

        return generated_plots

    def plot_training_curves(self, history_path: str = "results/training_history.json") -> Optional[str]:
        """Plot reward progression and loss during training."""
        if not os.path.exists(history_path):
            return None

        with open(history_path, "r") as f:
            history = json.load(f)

        episodes = [h["episode"] for h in history]
        train_rewards = [h["train_reward"] for h in history]
        losses = [h["loss"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(episodes, train_rewards, color="#2E86AB", label="Train Reward")
        # Moving average
        if len(train_rewards) >= 5:
            mv_avg = pd.Series(train_rewards).rolling(5, min_periods=1).mean()
            ax1.plot(episodes, mv_avg, color="#E63946", linestyle="--", label="Moving Avg (5 ep)")
        ax1.set_title("Training Reward vs Episode", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(episodes, losses, color="#F4A261", label="Training Loss")
        ax2.set_title("DQN Loss vs Episode", fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Huber Loss")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(self.plots_dir, "training_curves.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
