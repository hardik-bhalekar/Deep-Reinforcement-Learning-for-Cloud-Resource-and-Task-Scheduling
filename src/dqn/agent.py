"""Deep Q-Network (DQN) Agent for cloud task scheduling."""

from typing import Dict, Any, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dqn.network import QNetwork
from src.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """Double Deep Q-Network agent with action masking and experience replay."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Neural Networks (Online & Target)
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # 2. Optimizer & Huber (Smooth L1) Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        # 3. Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        self.update_count = 0

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, evaluate: bool = False) -> int:
        """Select a valid action using epsilon-greedy exploration with action masking."""
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return 0  # Fallback

        effective_epsilon = 0.0 if evaluate else self.epsilon

        # Epsilon-greedy exploration: choose uniformly among VALID actions only
        if np.random.rand() < effective_epsilon:
            return int(np.random.choice(valid_indices))

        # Exploitation: forward pass through Q-network with action masking
        self.online_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor).cpu().squeeze(0)

            # Mask invalid actions with a very large negative value
            masked_q = q_values.clone()
            for idx in range(self.action_dim):
                if not action_mask[idx]:
                    masked_q[idx] = -1e9

            action = int(torch.argmax(masked_q).item())
        self.online_net.train()
        return action

    def update(self) -> Optional[float]:
        """Perform a single Double-DQN gradient step from a sampled replay batch."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"].astype(np.float32)).to(self.device)
        next_action_masks = torch.BoolTensor(batch["next_action_masks"]).to(self.device)

        # 1. Current Q-values: Q(s, a)
        current_q = self.online_net(states).gather(1, actions).squeeze(1)

        # 2. Double DQN Target Calculation
        with torch.no_grad():
            # Online network selects the greedy action for next state (with action masking)
            next_q_online = self.online_net(next_states)
            next_q_online[~next_action_masks] = -1e9
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Target network evaluates the selected action
            next_q_target = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        # 3. Compute loss with gradient clipping safeguard
        loss = self.criterion(current_q, target_q)

        if torch.isnan(loss) or torch.isinf(loss):
            # Safeguard against gradient explosion or numerical instability
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 4. Periodic target network synchronization
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights and training state to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights and training state from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "update_count" in checkpoint:
            self.update_count = checkpoint["update_count"]
