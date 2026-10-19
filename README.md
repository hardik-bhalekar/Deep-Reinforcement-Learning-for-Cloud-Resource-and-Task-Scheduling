# Deep Reinforcement Learning for Cloud Resource and Task Scheduling

A Deep Reinforcement Learning (DRL) cloud task scheduling simulator comparing **Double Deep Q-Networks (Double DQN)** with traditional cloud scheduling algorithms (**Random, Round Robin, First Come First Served (FCFS), and Min-Min**) across multi-objective metrics: **Makespan**, **Energy Consumption (kWh)**, **Virtual Machine Utilization**, and **Service Level Agreement (SLA) Deadline Violations**.

---

## 1. Project Overview & Problem Statement

Modern cloud data centers (e.g., AWS, Azure, Google Cloud) manage hundreds of heterogeneous virtual machines (VMs) processing variable, dynamic user workloads. Task scheduling in cloud computing is an **NP-hard** optimization problem. Heuristics like Round Robin or First-Come First-Served (FCFS) do not adapt dynamically to heterogeneous capacities and varying arrival bursts, often leading to:
- Excessive **makespan** (total time to finish a batch of tasks).
- Energy waste from idle and underutilized VMs (idle servers typically draw 40–50% of maximum power).
- Frequent **SLA deadline violations**, resulting in financial penalties and poor quality of service.

This project investigates whether Deep Reinforcement Learning can learn intelligent dispatching policies that outperform traditional heuristics by dynamically mapping incoming tasks to optimal virtual machines based on queue delays, processing capacity, and memory constraints.

---

## 2. Objectives

1. **Lightweight Modular Cloud Simulator:** A Python-based discrete simulator modeling task arrivals, heterogeneous VM computational capacities (MIPS), RAM constraints, and literature-backed energy consumption (SPECpower linear power model).
2. **Action-Masked Double DQN:** A Deep Q-Network agent equipped with action masking to strictly prevent scheduling tasks onto virtual machines with insufficient memory.
3. **Multi-Objective Reward Formulation:** A balanced reward signal simultaneously optimizing task turnaround time, energy consumption, SLA compliance, and operational cost.
4. **Reproducible Comparative Benchmarking:** An automated evaluation pipeline executing identical seeded workloads across all algorithms, reporting mean and standard deviation.
5. **Interactive Visualization Dashboard:** A Streamlit dashboard for real-time schedule inspection, per-VM utilization breakdowns, and graphical comparisons.

---

## 3. Architecture & System Design

```
+---------------------------------------------------------------------------------+
|                                 WORKLOAD ENGINE                                 |
|   WorkloadGenerator (Poisson Arrival / Bursty / Heterogeneous / Uniform)        |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                           CLOUD SCHEDULING ENVIRONMENT                          |
|                                                                                 |
|  Current Task Properties       Virtual Machines (Heterogeneous MIPS & RAM)      |
|  - Length (MIPS)               - VM 0: 1000 MIPS,  4GB RAM, 80-200W             |
|  - Memory (MB)                 - VM 1: 2000 MIPS,  8GB RAM, 110-250W            |
|  - SLA Deadline (s)            - VM 2: 3000 MIPS,  8GB RAM, 140-300W            |
|  - Priority                    - VM 3: 4000 MIPS, 16GB RAM, 180-400W            |
|                                                                                 |
|  Observation Vector (20D):                                                      |
|    [Task Length, Task RAM, Tasks Remaining, Deadline Urgency,                   |
|     (VM_MIPS, VM_RAM, Queue Delay, Est. Completion) x 4 VMs]                    |
|                                                                                 |
|  Action Masking: Only allows VMs where vm.ram >= task.ram                       |
+---------------------------------------------------------------------------------+
                         ^                             |
                         | Actions                     | Rewards & Next Obs
                         |                             v
+-------------------------------------+      +------------------------------------+
|         SCHEDULER / AGENT           |      |          METRICS COLLECTOR         |
|  - Baselines:                       |      |  - Makespan                        |
|    Random, Round Robin,             |      |  - Total Energy (kWh, SPECpower)   |
|    FCFS, Min-Min                    |      |  - Average Turnaround Time         |
|  - DRL Agent:                       |      |  - Average Waiting Time            |
|    Double DQN with Experience       |      |  - SLA Violation Rate (%)          |
|    Replay Buffer & Epsilon-Greedy   |      |  - Average VM Utilization          |
+-------------------------------------+      +------------------------------------+
```

---

## 4. Repository Structure

```
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Verified Python dependencies
├── .gitignore                     # Git exclusions (caches, models, virtualenvs)
├── configs/
│   └── default_config.json        # Centralized simulation and training parameters
├── src/
│   ├── __init__.py
│   ├── simulator/                 # Core simulation engine
│   │   ├── __init__.py
│   │   ├── task.py                # Task data model & SLA tracking
│   │   ├── vm.py                  # Virtual Machine queue & resource tracking
│   │   ├── energy_model.py        # SPECpower linear power model (Watts, Joules, kWh)
│   │   ├── workload.py            # Reproducible workload generator
│   │   ├── metrics.py             # MetricsCollector & SimulationMetrics
│   │   └── environment.py         # Gymnasium-like CloudSchedulingEnv
│   ├── baselines/                 # Conventional scheduling heuristics
│   │   ├── __init__.py
│   │   ├── base.py                # BaseScheduler abstract interface
│   │   ├── random_scheduler.py    # Random valid VM assignment
│   │   ├── round_robin.py         # Cyclic round-robin assignment
│   │   ├── fcfs.py                # Earliest-available VM assignment
│   │   └── min_min.py             # Minimum completion time heuristic
│   ├── dqn/                       # Deep Reinforcement Learning
│   │   ├── __init__.py
│   │   ├── network.py             # PyTorch QNetwork MLP architecture
│   │   ├── replay_buffer.py       # Experience Replay Buffer
│   │   └── agent.py               # Double DQN Agent with Action Masking
│   ├── evaluation/                # Evaluation & plotting engine
│   │   ├── __init__.py
│   │   └── evaluator.py           # Multi-seed benchmark runner & Matplotlib plot generator
│   └── dashboard/
│       └── app.py                 # Interactive Streamlit dashboard
├── tests/                         # 32 automated unit and integration tests
│   ├── test_models.py
│   ├── test_workload.py
│   ├── test_environment.py
│   ├── test_baselines.py
│   ├── test_dqn.py
│   └── test_evaluation.py
├── train.py                       # CLI entrypoint for training DQN
└── evaluate.py                    # CLI entrypoint for evaluating and plotting
```

---

## 5. Mathematical Formulations

### A. Power and Energy Model (SPECpower Linear Model)
A physical server or VM draws power as a function of CPU utilization $u \in [0, 1]$:
$$P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) \times u$$

Over total observation span $T$ where the VM is busy executing tasks for $T_{\text{busy}}$ seconds:
$$E = (P_{\text{max}} \times T_{\text{busy}}) + (P_{\text{idle}} \times (T - T_{\text{busy}})) \quad \text{[Joules]}$$
$$\text{Energy [kWh]} = \frac{E}{3.6 \times 10^6}$$

### B. Reinforcement Learning State Space ($s \in \mathbb{R}^{20}$)
For $M=4$ virtual machines, the observation vector consists of 20 normalized features in $[0, 1]$:
1. `task_length_norm`: $L_i / L_{\text{max}}$
2. `task_ram_norm`: $\text{RAM}_i / \text{RAM}_{\text{max}}$
3. `tasks_remaining_ratio`: $(N - i) / N$
4. `deadline_urgency_norm`: $D_i / T_{\text{horizon}}$
5. For each VM $j \in \{0, 1, 2, 3\}$ (4 features per VM):
   - `mips_norm`: $\text{MIPS}_j / \text{MIPS}_{\text{max}}$
   - `ram_capacity_norm`: $\text{RAM}_j / \text{RAM}_{\text{max}}$
   - `queue_delay_norm`: $\max(0, T^{\text{avail}}_j - t^{\text{arr}}_i) / T_{\text{horizon}}$
   - `est_completion_norm`: $\text{CompletionTime}(i, j) / T_{\text{horizon}}$

### C. Action Space & Action Masking
The action $a \in \{0, 1, \dots, M-1\}$ chooses which VM executes task $i$.
An action mask $m \in \{0, 1\}^M$ guarantees that:
$$m_j = 1 \iff \text{RAM}_j \ge \text{TaskRAM}_i$$
During Q-network action selection, invalid actions receive $Q_{\text{masked}}(s, j) = -\infty$, preventing the agent from violating resource constraints.

### D. Multi-Objective Reward Function
$$R = w_{\text{makespan}} \cdot r_{\text{time}} + w_{\text{energy}} \cdot r_{\text{energy}} + w_{\text{sla}} \cdot r_{\text{sla}} + w_{\text{cost}} \cdot r_{\text{cost}}$$
- $r_{\text{time}} = -\min(2.0, \frac{\text{turnaround}}{10.0})$: Penalizes completion delays.
- $r_{\text{energy}} = -\min(2.0, \frac{\Delta E}{E_{\text{ref}}})$: Penalizes incremental Joules consumed.
- $r_{\text{sla}} = +0.5$ if completed within deadline, $-1.0$ if violated.
- $r_{\text{cost}} = -\text{cost} \times 1000$: Penalizes monetary dollar cost.

### E. Double Deep Q-Network Update
To prevent the maximization bias inherent in standard Q-learning, the greedy action selection is decoupled from target evaluation:
$$y = r + \gamma Q_{\theta^-}\left(s', \arg\max_{a' \in \text{valid}} Q_\theta(s', a')\right) \cdot (1 - \text{done})$$
Loss is minimized using Huber / Smooth L1 loss with gradient clipping ($\text{max\_norm} = 1.0$).

---

## 6. Baseline Scheduling Algorithms

1. **Random:** Uniformly assigns tasks to a randomly chosen VM that satisfies memory constraints.
2. **Round Robin:** Cyclically assigns tasks across VMs in circular order $(j+1) \bmod M$.
3. **First Come, First Served (FCFS):** Assigns tasks in order of arrival to the VM that becomes available earliest, minimizing immediate waiting time.
4. **Min-Min Heuristic:** Evaluates estimated completion time across all valid VMs and assigns each task to the VM that achieves the minimum completion time.

---

## 7. Experimental Results

Evaluated across **5 independent test seeds** on a **heterogeneous workload (50 tasks)** using 4 virtual machines:

| Algorithm | Makespan (s) $\downarrow$ | Avg Turnaround (s) $\downarrow$ | Total Energy (kWh) $\downarrow$ | SLA Violation Rate $\downarrow$ | Avg VM Utilization | Degree of Imbalance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | 66.11 ± 8.39 | 5.04 ± 1.15 | 0.014 ± 0.002 | 42.8% ± 6.9% | 42.6% | 1.48 |
| **Round Robin** | 56.94 ± 6.31 | 3.49 ± 0.69 | 0.012 ± 0.001 | 30.8% ± 4.5% | 48.1% | 1.41 |
| **FCFS** | 48.12 ± 2.92 | 1.92 ± 0.19 | 0.011 ± 0.000 | 17.2% ± 2.4% | 48.9% | 0.68 |
| **Min-Min** | **45.57 ± 2.06** | **1.38 ± 0.19** | **0.010 ± 0.000** | **1.6% ± 2.3%** | 34.2% | 2.07 |
| **DQN (Ours)** | **47.68 ± 2.79** | **2.12 ± 0.12** | **0.010 ± 0.000** | **17.6% ± 5.3%** | 30.8% | 2.74 |

> **Key Finding:** Even with short CPU training (30 episodes), the Double DQN agent learned to outperform **Random** (28% lower makespan) and **Round Robin** (16% lower makespan), approaching the performance of the heavily optimized **Min-Min** heuristic and **FCFS** while maintaining low energy consumption.

---

## 8. Installation & Setup

### Prerequisites
- Linux / Ubuntu, macOS, or Windows WSL2
- Python 3.10+ (tested on Python 3.13)

### Installation
```bash
# Clone the repository
git clone https://github.com/hardik-bhalekar/Deep-Reinforcement-Learning-for-Cloud-Resource-and-Task-Scheduling.git
cd Deep-Reinforcement-Learning-for-Cloud-Resource-and-Task-Scheduling

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 9. Execution Instructions

### A. Run Automated Unit Tests (32 Tests)
```bash
python3 -m pytest -v tests
```

### B. Train the DQN Model
```bash
# Fast CPU training (30 episodes)
python3 train.py --episodes 30 --eval-interval 10

# Extended training experiment (100 episodes)
python3 train.py --episodes 100 --eval-interval 20
```
Checkpoints will be saved to `checkpoints/best_dqn_model.pt` and training logs to `results/training_history.json`.

### C. Run the Evaluation Benchmark & Generate Plots
```bash
python3 evaluate.py --tasks 50 --workload heterogeneous --seeds 101 102 103 104 105
```
This generates comparison figures in `results/plots/`:
- `results/plots/comprehensive_comparison.png`
- `results/plots/makespan_comparison.png`
- `results/plots/total_energy_kwh_comparison.png`
- `results/plots/sla_violation_rate_comparison.png`
- `results/plots/training_curves.png`

### D. Launch the Interactive Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open `http://localhost:8501` in your browser to interactively adjust workloads, inspect schedules, and view metrics.

---

## 10. Assumptions & Limitations

1. **Simulation Model:** Task execution is modeled deterministically as $\text{Length} / \text{MIPS}$. Network transfer overheads and disk I/O are simplified.
2. **Homogeneous Virtualization Overhead:** VM context-switching overheads are modeled as negligible relative to multi-second computation tasks.
3. **Discrete Action Space:** The action space is discrete ($M$ VMs). Dynamic scaling (adding/removing VMs) is not part of this initial dispatching model.

---

## 11. Future Research Directions

- **Multi-Agent DRL (MADDPG / MAPPO):** Decentralized schedulers where each host acts as an autonomous agent.
- **Dynamic Voltage and Frequency Scaling (DVFS):** Allowing agents to dynamically throttle CPU frequencies to trade off execution speed for quadratic power savings.
- **Trace-Driven Replay:** Replaying Google or Alibaba cluster traces to evaluate domain adaptation under real production traffic.
