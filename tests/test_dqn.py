"""Unit tests for Deep Q-Network (DQN) components."""

import unittest
import os
import tempfile
import numpy as np
import torch
from src.dqn.network import QNetwork
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.agent import DQNAgent


class TestQNetwork(unittest.TestCase):
    def test_forward_pass_dimensions(self):
        state_dim = 20
        action_dim = 4
        hidden_dim = 64
        net = QNetwork(state_dim, action_dim, hidden_dim)

        # Single state forward
        state = torch.randn(state_dim)
        q_vals = net(state)
        self.assertEqual(q_vals.shape, (action_dim,))

        # Batch forward
        batch = torch.randn(16, state_dim)
        batch_q_vals = net(batch)
        self.assertEqual(batch_q_vals.shape, (16, action_dim))


class TestReplayBuffer(unittest.TestCase):
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100, state_dim=5, action_dim=2)
        self.assertEqual(len(buf), 0)

        for i in range(10):
            buf.push(
                state=np.full(5, i, dtype=np.float32),
                action=i % 2,
                reward=float(i),
                next_state=np.full(5, i + 1, dtype=np.float32),
                done=(i == 9),
                next_action_mask=np.array([True, False]),
            )

        self.assertEqual(len(buf), 10)
        batch = buf.sample(batch_size=4)
        self.assertEqual(batch["states"].shape, (4, 5))
        self.assertEqual(batch["actions"].shape, (4,))
        self.assertEqual(batch["rewards"].shape, (4,))
        self.assertEqual(batch["next_states"].shape, (4, 5))
        self.assertEqual(batch["dones"].shape, (4,))
        self.assertEqual(batch["next_action_masks"].shape, (4, 2))


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=32,
            lr=0.01,
            batch_size=8,
            buffer_capacity=100,
        )

    def test_action_selection_with_masking(self):
        state = np.random.randn(10).astype(np.float32)
        # Action mask: only action 1 is valid
        mask = np.array([False, True, False])

        # Test both evaluation (greedy) and exploration
        for _ in range(5):
            action_eval = self.agent.select_action(state, mask, evaluate=True)
            self.assertEqual(action_eval, 1)

            action_explore = self.agent.select_action(state, mask, evaluate=False)
            self.assertEqual(action_explore, 1)

    def test_update_step(self):
        # Fill buffer past batch size
        for _ in range(20):
            self.agent.replay_buffer.push(
                state=np.random.randn(10).astype(np.float32),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(10).astype(np.float32),
                done=False,
                next_action_mask=np.array([True, True, True]),
            )

        loss = self.agent.update()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "model.pt")
            self.agent.epsilon = 0.42
            self.agent.save_checkpoint(chk_path)
            self.assertTrue(os.path.exists(chk_path))

            new_agent = DQNAgent(state_dim=10, action_dim=3, hidden_dim=32)
            new_agent.load_checkpoint(chk_path)
            self.assertAlmostEqual(new_agent.epsilon, 0.42)

            # Check parameter values match
            for p1, p2 in zip(self.agent.online_net.parameters(), new_agent.online_net.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
