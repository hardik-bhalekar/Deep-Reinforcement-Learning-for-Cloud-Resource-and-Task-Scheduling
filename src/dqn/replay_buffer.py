"""Experience Replay Buffer for Deep Q-Learning."""

from typing import Tuple, Dict
import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for storing transition tuples.

    Stores: (state, action, reward, next_state, done, next_action_mask)
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.next_action_masks = np.ones((self.capacity, self.action_dim), dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        """Add a transition tuple into the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.next_action_masks[self.ptr] = next_action_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions uniformly at random."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {self.size}")

        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "next_action_masks": self.next_action_masks[indices],
        }

    def __len__(self) -> int:
        return self.size
