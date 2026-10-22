"""Training pipeline for Deep Q-Network on Cloud Scheduling Environment."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.simulator.vm import VirtualMachine
from src.simulator.energy_model import EnergyModel
from src.simulator.metrics import MetricsCollector
from src.dqn.agent import DQNAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON experiment configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def build_environment(config: Dict[str, Any]) -> CloudSchedulingEnv:
    """Instantiate simulation environment from configuration parameters."""
    sim_cfg = config.get("simulation", {})
    vm_profiles = config.get("vm_profiles", [])
    reward_weights = config.get("reward_weights", {})

    vms = []
    for idx, p in enumerate(vm_profiles):
        vms.append(
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
        )

    workload = WorkloadGenerator(
        num_tasks=int(sim_cfg.get("num_tasks", 50)),
        workload_type=str(sim_cfg.get("workload_type", "heterogeneous")),
        seed=int(sim_cfg.get("seed", 42)),
        length_min=float(sim_cfg.get("task_length_min", 1000.0)),
        length_max=float(sim_cfg.get("task_length_max", 10000.0)),
        ram_min=float(sim_cfg.get("task_ram_min", 512.0)),
        ram_max=float(sim_cfg.get("task_ram_max", 4096.0)),
        arrival_rate=float(sim_cfg.get("arrival_rate", 1.5)),
        deadline_slack=float(sim_cfg.get("deadline_slack_factor", 1.5)),
    )

    env = CloudSchedulingEnv(
        vms=vms,
        workload_generator=workload,
        reward_weights=reward_weights,
    )
    return env


def evaluate_agent(env: CloudSchedulingEnv, agent: DQNAgent, eval_tasks: List) -> Dict[str, Any]:
    """Evaluate current policy greedily (epsilon=0) on fixed evaluation workload."""
    obs, info = env.reset(tasks=eval_tasks)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(obs, info["action_mask"], evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    metrics = MetricsCollector.calculate_metrics(env.tasks, env.vms, scheduler_name="DQN-Eval")
    res = metrics.to_dict()
    res["eval_reward"] = round(total_reward, 3)
    return res


def train(
    config_path: str = "configs/default_config.json",
    num_episodes: int = 100,
    save_dir: str = "checkpoints",
    results_dir: str = "results",
    eval_interval: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Main training routine for Deep Q-Network cloud scheduler."""
    logger.info("Initializing DRL training pipeline...")
    config = load_config(config_path)

    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = build_environment(config)
    dqn_cfg = config.get("dqn", {})

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(dqn_cfg.get("hidden_dim", 128)),
        lr=float(dqn_cfg.get("learning_rate", 0.001)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        epsilon_start=float(dqn_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(dqn_cfg.get("epsilon_end", 0.05)),
        epsilon_decay=float(dqn_cfg.get("epsilon_decay", 0.995)),
        buffer_capacity=int(dqn_cfg.get("buffer_capacity", 10000)),
        batch_size=int(dqn_cfg.get("batch_size", 64)),
        target_update_freq=int(dqn_cfg.get("target_update_freq", 10)),
    )

    # Generate fixed evaluation workload for fair tracking across training
    eval_workload_gen = WorkloadGenerator(
        num_tasks=int(config.get("simulation", {}).get("num_tasks", 50)),
        workload_type=str(config.get("simulation", {}).get("workload_type", "heterogeneous")),
        seed=seed + 9999,
    )
    eval_tasks = eval_workload_gen.generate()

    history: List[Dict[str, Any]] = []
    best_eval_reward = -float("inf")

    logger.info(
        f"Starting training: {num_episodes} episodes | State Dim: {env.state_dim} | Action Dim (VMs): {env.action_dim}"
    )

    for episode in range(1, num_episodes + 1):
        # Fresh workload per episode using seeded random state
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        step_losses = []
        done = False

        while not done:
            action = agent.select_action(obs, info["action_mask"], evaluate=False)
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            agent.replay_buffer.push(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=terminated,
                next_action_mask=next_info["action_mask"],
            )

            loss = agent.update()
            if loss is not None:
                step_losses.append(loss)

            obs = next_obs
            info = next_info
            episode_reward += reward
            done = terminated or truncated

        agent.decay_epsilon()
        avg_loss = float(np.mean(step_losses)) if step_losses else 0.0

        # Periodic Evaluation
        eval_metrics = None
        if episode % eval_interval == 0 or episode == num_episodes:
            eval_metrics = evaluate_agent(env, agent, eval_tasks)
            eval_rew = eval_metrics["eval_reward"]
            eval_makespan = eval_metrics["makespan"]
            eval_energy = eval_metrics["total_energy_kwh"]
            eval_sla = eval_metrics["sla_violation_rate"]

            logger.info(
                f"Episode {episode:3d}/{num_episodes} | Train Rew: {episode_reward:7.2f} | "
                f"Eval Rew: {eval_rew:7.2f} | Makespan: {eval_makespan:6.2f}s | "
                f"Energy: {eval_energy:.5f} kWh | SLA Viol: {eval_sla * 100:4.1f}% | "
                f"Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
            )

            if eval_rew > best_eval_reward:
                best_eval_reward = eval_rew
                best_model_path = os.path.join(save_dir, "best_dqn_model.pt")
                agent.save_checkpoint(best_model_path)
                logger.info(f"--> Saved new best model to {best_model_path}")
        else:
            if episode % 5 == 0:
                logger.info(
                    f"Episode {episode:3d}/{num_episodes} | Train Rew: {episode_reward:7.2f} | "
                    f"Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
                )

        ep_record = {
            "episode": episode,
            "train_reward": round(episode_reward, 3),
            "loss": round(avg_loss, 5),
            "epsilon": round(agent.epsilon, 4),
        }
        if eval_metrics:
            ep_record["eval"] = eval_metrics

        history.append(ep_record)

    # Save final model and history
    final_model_path = os.path.join(save_dir, "final_dqn_model.pt")
    agent.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    history_file = os.path.join(results_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history log to {history_file}")

    return {
        "final_model_path": final_model_path,
        "best_model_path": os.path.join(save_dir, "best_dqn_model.pt"),
        "history_file": history_file,
        "best_eval_reward": best_eval_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DQN Agent for Cloud Task Scheduling")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config JSON")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=20, help="Evaluation interval in episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metrics logs")
    args = parser.parse_args()

    train(
        config_path=args.config,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
