"""Deep Q-Network (QNetwork) neural network architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Deep Q-Network approximator estimating state-action values Q(s, a).

    Architecture:
        Input Layer: state_dim
        Hidden Layer 1: hidden_dim (ReLU)
        Hidden Layer 2: hidden_dim (ReLU)
        Output Layer: action_dim (Linear Q-values for each VM)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply He/Kaiming initialization suitable for ReLU activations."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given state vector(s).

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Tensor of shape (batch_size, action_dim) or (action_dim,) containing Q-values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
