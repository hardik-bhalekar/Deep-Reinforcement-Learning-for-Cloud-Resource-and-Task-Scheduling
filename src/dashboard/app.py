"""Streamlit Interactive Dashboard for Cloud Resource & Task Scheduling."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.simulator.environment import CloudSchedulingEnv
from src.simulator.workload import WorkloadGenerator
from src.simulator.metrics import MetricsCollector
from src.baselines.random_scheduler import RandomScheduler
from src.baselines.round_robin import RoundRobinScheduler
from src.baselines.fcfs import FCFSScheduler
from src.baselines.min_min import MinMinScheduler
from src.dqn.agent import DQNAgent

st.set_page_config(
    page_title="Cloud Scheduling DRL Simulator",
    page_icon="☁️",
    layout="wide",
)

st.title("☁️ Deep Reinforcement Learning for Cloud Resource & Task Scheduling")
st.markdown(
    """
    This interactive dashboard demonstrates how **Deep Reinforcement Learning (Double DQN)** compares 
    against conventional cloud scheduling heuristics (**Random, Round Robin, FCFS, Min-Min**) 
    in optimizing **Makespan**, **Energy Consumption**, and **SLA Deadline Violations**.
    """
)


@st.cache_resource
def get_environment():
    config_path = "configs/default_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    return CloudSchedulingEnv()


@st.cache_resource
def load_dqn_agent(model_path: str, state_dim: int, action_dim: int):
    if not os.path.exists(model_path):
        return None
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=128)
    agent.load_checkpoint(model_path)
    return agent


# Sidebar configuration
st.sidebar.header("🛠️ Workload & Experiment Setup")
num_tasks = st.sidebar.slider("Number of Tasks", min_value=10, max_value=150, value=50, step=10)
workload_type = st.sidebar.selectbox(
    "Workload Distribution",
    options=["heterogeneous", "bursty", "poisson_arrival", "uniform"],
    index=0,
    help="Heterogeneous mixes short interactive and long compute-heavy jobs. Bursty clusters arrivals simultaneously.",
)
seed = st.sidebar.number_input("Random Seed (Reproducibility)", min_value=0, max_value=99999, value=42)

model_path = st.sidebar.text_input("DQN Checkpoint Path", value="checkpoints/best_dqn_model.pt")

env = get_environment()
dqn_agent = load_dqn_agent(model_path, env.state_dim, env.action_dim)
if dqn_agent:
    st.sidebar.success("✅ Trained DQN Model Loaded")
else:
    st.sidebar.warning(f"⚠️ Model checkpoint '{model_path}' not found. Train first with train.py")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Comparative Benchmark", "🔍 Detailed Execution Analysis", "🧠 DRL Architecture & Theory"])

with tab1:
    st.subheader("Comparative Benchmark on Identical Workload")
    st.markdown(
        f"Evaluating algorithms on **{num_tasks} tasks** under the **{workload_type}** workload distribution (Seed: `{seed}`)."
    )

    if st.button("🚀 Run Comparative Benchmark", use_container_width=True):
        workload_gen = WorkloadGenerator(
            num_tasks=num_tasks,
            workload_type=workload_type,
            seed=seed,
        )

        schedulers = {
            "Random": RandomScheduler(seed=seed),
            "Round Robin": RoundRobinScheduler(),
            "FCFS": FCFSScheduler(),
            "Min-Min": MinMinScheduler(),
        }

        results = []
        for name, scheduler in schedulers.items():
            tasks = workload_gen.generate()
            env.reset(tasks=tasks)
            done = False
            while not done:
                action = scheduler.select_action(env)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            m = MetricsCollector.calculate_metrics(env.tasks, env.vms, scheduler_name=name)
            results.append(m.to_dict())

        if dqn_agent is not None:
            tasks = workload_gen.generate()
            obs, info = env.reset(tasks=tasks)
            done = False
            while not done:
                action = dqn_agent.select_action(obs, info["action_mask"], evaluate=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            m = MetricsCollector.calculate_metrics(env.tasks, env.vms, scheduler_name="DQN")
            results.append(m.to_dict())

        df_res = pd.DataFrame(results)
        st.session_state["benchmark_df"] = df_res

    if "benchmark_df" in st.session_state:
        df_res = st.session_state["benchmark_df"]

        # Display table
        st.dataframe(
            df_res[
                [
                    "scheduler_name",
                    "makespan",
                    "avg_turnaround_time",
                    "avg_waiting_time",
                    "total_energy_kwh",
                    "avg_vm_utilization",
                    "sla_violation_rate",
                    "total_cost",
                ]
            ],
            use_container_width=True,
        )

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.bar(df_res["scheduler_name"], df_res["makespan"], color="#4A90E2", edgecolor="black")
            ax1.set_title("Makespan (Lower is Better)", fontweight="bold")
            ax1.set_ylabel("Seconds")
            ax1.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(df_res["scheduler_name"], df_res["total_energy_kwh"], color="#50E3C2", edgecolor="black")
            ax2.set_title("Total Energy Consumption (Lower is Better)", fontweight="bold")
            ax2.set_ylabel("kWh")
            ax2.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.bar(
                df_res["scheduler_name"],
                df_res["sla_violation_rate"] * 100,
                color="#E63946",
                edgecolor="black",
            )
            ax3.set_title("SLA Deadline Violation Rate (%)", fontweight="bold")
            ax3.set_ylabel("Violation %")
            ax3.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig3)

        with col4:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.bar(
                df_res["scheduler_name"],
                df_res["avg_vm_utilization"] * 100,
                color="#F5A623",
                edgecolor="black",
            )
            ax4.set_title("Average VM Utilization (%)", fontweight="bold")
            ax4.set_ylabel("Utilization %")
            ax4.grid(axis="y", linestyle="--", alpha=0.7)
            st.pyplot(fig4)

with tab2:
    st.subheader("Single Scheduler Deep-Dive & Timeline")
    selected_algo = st.selectbox("Select Algorithm to Inspect", ["DQN", "Min-Min", "FCFS", "Round Robin", "Random"])

    if st.button("Inspect Schedule", use_container_width=True):
        workload_gen = WorkloadGenerator(num_tasks=num_tasks, workload_type=workload_type, seed=seed)
        tasks = workload_gen.generate()
        obs, info = env.reset(tasks=tasks)
        done = False

        if selected_algo == "DQN" and dqn_agent:
            while not done:
                action = dqn_agent.select_action(obs, info["action_mask"], evaluate=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        else:
            sched_map = {
                "Random": RandomScheduler(seed=seed),
                "Round Robin": RoundRobinScheduler(),
                "FCFS": FCFSScheduler(),
                "Min-Min": MinMinScheduler(),
            }
            sched = sched_map.get(selected_algo, MinMinScheduler())
            while not done:
                action = sched.select_action(env)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        metrics = MetricsCollector.calculate_metrics(env.tasks, env.vms, scheduler_name=selected_algo)

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Makespan", f"{metrics.makespan:.2f} s")
        kpi2.metric("Total Energy", f"{metrics.total_energy_kwh:.5f} kWh")
        kpi3.metric("SLA Violations", f"{metrics.sla_violation_rate * 100:.1f} %")
        kpi4.metric("Avg Utilization", f"{metrics.avg_vm_utilization * 100:.1f} %")

        # Per VM breakdown
        st.markdown("#### Virtual Machine Resource Allocation")
        vm_data = [vm.to_dict() for vm in env.vms]
        df_vms = pd.DataFrame(vm_data)
        st.dataframe(df_vms, use_container_width=True)

        # Task assignments table
        st.markdown("#### Task Execution Timeline")
        task_data = [t.to_dict() for t in env.tasks[:30]]
        st.dataframe(pd.DataFrame(task_data), use_container_width=True)

with tab3:
    st.subheader("Deep Reinforcement Learning Architecture & Training Progression")

    history_path = "results/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        episodes = [h["episode"] for h in history]
        rewards = [h["train_reward"] for h in history]
        losses = [h["loss"] for h in history]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(episodes, rewards, color="#2E86AB")
            ax.set_title("Training Reward vs Episode", fontweight="bold")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Episode Reward")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(episodes, losses, color="#F4A261")
            ax.set_title("Huber Loss vs Episode", fontweight="bold")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
    else:
        st.info("No training history log found at results/training_history.json. Run `python train.py` to populate.")

    st.markdown(
        r"""
        ### Reinforcement Learning Formulation:
        - **State Representation ($s \in \mathbb{R}^{20}$):**
          - Current Task: Normalized length, memory demand, deadline urgency, remaining task fraction.
          - Per-VM (for 4 VMs): Normalized MIPS, RAM capacity, queue delay, and estimated finish timestamp.
        - **Action Space ($a \in \{0, 1, 2, 3\}$):** Selection of destination virtual machine, constrained by action masking (memory compatibility).
        - **Reward Function:** Multi-objective penalty combining turnaround time, energy consumed, SLA deadline bonus/penalty, and resource cost:
          $$R = w_{\text{makespan}} r_{\text{time}} + w_{\text{energy}} r_{\text{energy}} + w_{\text{sla}} r_{\text{sla}} + w_{\text{cost}} r_{\text{cost}}$$
        - **Optimization:** Double Deep Q-Network (Double DQN) with Experience Replay Buffer and Huber Loss to combat Q-value overestimation.
        """
    )
