"""Evaluation CLI entrypoint for Cloud Scheduling."""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.simulator.vm import VirtualMachine
from src.simulator.energy_model import EnergyModel
from src.dqn.agent import DQNAgent
from src.evaluation.evaluator import BenchmarkEvaluator


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def build_environment(config):
    vm_profiles = config.get("vm_profiles", [])
    reward_weights = config.get("reward_weights", {})

    vms = [
        VirtualMachine(
            vm_id=idx,
            mips=float(p["mips"]),
            ram_mb=float(p["ram_mb"]),
            cost_per_hour=float(p.get("cost_per_hour", 0.10)),
            energy_model=EnergyModel(
                idle_power_w=float(p.get("idle_power_w", 100.0)),
                max_power_w=float(p.get("max_power_w", 250.0)),
            ),
        )
        for idx, p in enumerate(vm_profiles)
    ]
    return CloudSchedulingEnv(vms=vms, reward_weights=reward_weights)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cloud Scheduling Algorithms")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config file")
    parser.add_argument("--model", type=str, default="checkpoints/best_dqn_model.pt", help="Path to trained DQN model")
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks per test run")
    parser.add_argument("--workload", type=str, default="heterogeneous", help="Workload distribution type")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105], help="Evaluation seeds")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_environment(config)

    dqn_agent = None
    if os.path.exists(args.model):
        print(f"Loading trained DQN model from {args.model}...")
        dqn_cfg = config.get("dqn", {})
        dqn_agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=int(dqn_cfg.get("hidden_dim", 128)),
        )
        dqn_agent.load_checkpoint(args.model)
    else:
        print(f"Warning: Model checkpoint '{args.model}' not found. Evaluating baseline heuristics only.")

    evaluator = BenchmarkEvaluator(
        env=env,
        dqn_agent=dqn_agent,
        test_seeds=args.seeds,
        results_dir=args.results_dir,
    )

    print(f"\nRunning benchmark across {len(args.seeds)} test seeds ({args.tasks} tasks, {args.workload} workload)...")
    results = evaluator.run_benchmark(num_tasks=args.tasks, workload_type=args.workload)

    # Generate plots
    print("Generating comparison charts...")
    plots = evaluator.generate_comparison_plots(results)
    train_plot = evaluator.plot_training_curves(os.path.join(args.results_dir, "training_history.json"))
    if train_plot:
        plots.append(train_plot)

    for p in plots:
        print(f"  Saved plot: {p}")

    # Print summary table
    summary_csv = os.path.join(args.results_dir, "benchmark_summary.csv")
    df = pd.read_csv(summary_csv)
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY (Mean ± Std)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
