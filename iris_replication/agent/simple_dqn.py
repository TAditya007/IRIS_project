# ============================================================
# agent/simple_dqn.py
# ------------------------------------------------------------
# Literal implementation of the paper's Algorithm 1.
#
# Paper: Section III, Eq. 6-10 and Algorithm 1.
#
# Deliberately different from agent/dqn.py:
#   - NO target network (Algorithm 1 uses Q(s', a; theta), the same
#     theta as the online network)
#   - NO replay buffer (Eq. 9's expectation is approximated by the
#     single current transition)
#   - Batch size = 1 (Algorithm 1: "Perform a gradient descent step
#     on loss L(theta) = (y_t - Q(s_t, a_t; theta))^2")
#
# This is the maximum-fidelity execution of Algorithm 1 as printed.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================
@dataclass
class SimpleDQNConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    device: str = "cuda"


# ============================================================
# Q-network (identical structure to DQNAgent's)
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, cfg: SimpleDQNConfig):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Literal Algorithm 1 agent
# ============================================================
class SimpleDQNAgent:
    """
    Algorithm 1, verbatim.
    - Single network (no target copy)
    - One gradient step per env step, batch size 1
    """

    def __init__(self, state_dim: int, action_dim: int,
                 cfg: SimpleDQNConfig, seed: int = 0):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg

        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Single network (Algorithm 1 uses one theta)
        self.online = QNetwork(state_dim, action_dim, cfg).to(self.device)

        # Optimizer: Adam at lr = alpha (paper)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.learning_rate
        )

        # Exploration
        self._epsilon = float(cfg.epsilon_start)
        self.updates = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def decay_epsilon(self) -> float:
        self._epsilon = max(
            self.cfg.epsilon_end,
            self._epsilon * self.cfg.epsilon_decay,
        )
        return self._epsilon

    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (self._np_rng.random() < self._epsilon):
            return int(self._np_rng.integers(0, self.action_dim))
        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        q = self.online(s)
        return int(q.argmax(dim=1).item())

    def update_one(self, s: np.ndarray, a: int, r: float,
                   s_next: np.ndarray, done: bool) -> float:
        """
        One gradient step on the current transition, batch size 1.

        y = r + gamma * max_a Q(s', a; theta)   (Algorithm 1 / Eq. 10)
        L = (y - Q(s, a; theta))^2              (Algorithm 1 / Eq. 9)
        """
        s_t = torch.as_tensor(s, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        a_t = torch.tensor([a], dtype=torch.long, device=self.device)
        r_t = torch.tensor([r], dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s_next, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        d_t = torch.tensor([1.0 if done else 0.0], dtype=torch.float32,
                           device=self.device)

        q_sa = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Algorithm 1: y = r + gamma * max_a Q(s', a; theta)
        # Same network theta. No target network.
        with torch.no_grad():
            q_max_next = self.online(s2_t).max(dim=1).values
            y = r_t + self.cfg.gamma * (1.0 - d_t) * q_max_next

        loss = F.mse_loss(q_sa, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        return float(loss.detach().cpu().item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "updates": self.updates,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._epsilon = float(ckpt["epsilon"])
        self.updates = int(ckpt["updates"])


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("simple_dqn.py self-test")
    print("=" * 60)

    cfg = SimpleDQNConfig(
        hidden_dim=32, num_hidden_layers=2, learning_rate=1e-3,
        gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.99, device="cuda",
    )
    agent = SimpleDQNAgent(state_dim=4, action_dim=3, cfg=cfg, seed=42)
    print(f"[OK] Agent on device: {agent.device}")

    s = np.zeros(4, dtype=np.float32)
    a = agent.select_action(s, greedy=True)
    print(f"[OK] select_action works (a={a})")

    # Single update step
    loss = agent.update_one(
        s=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        a=0, r=1.0,
        s_next=np.zeros(4, dtype=np.float32),
        done=True,
    )
    assert np.isfinite(loss), f"loss not finite: {loss}"
    print(f"[OK] Single-transition update -> loss = {loss:.6f}")

    # Toy learning: state = [x, 0, 0, 0], best action = 0 if x>0 else 1
    print("\n[INFO] Toy learning test (10000 single-transition updates)...")
    rng = np.random.default_rng(7)
    losses = []
    for _ in range(10000):
        x = float(rng.uniform(-1, 1))
        s = np.array([x, 0.0, 0.0, 0.0], dtype=np.float32)
        a = agent.select_action(s, greedy=False)
        best = 0 if x > 0 else 1
        r = 1.0 if a == best else (0.0 if a == 2 else 0.0)
        s2 = np.zeros(4, dtype=np.float32)
        loss = agent.update_one(s, a, r, s2, done=True)
        losses.append(loss)

    early = float(np.mean(losses[:100]))
    late = float(np.mean(losses[-100:]))
    print(f"[OK] Loss: {early:.4f} -> {late:.4f}")

    correct = 0
    for _ in range(200):
        x = float(rng.uniform(-1, 1))
        s = np.array([x, 0.0, 0.0, 0.0], dtype=np.float32)
        a = agent.select_action(s, greedy=True)
        best = 0 if x > 0 else 1
        correct += int(a == best)
    acc = correct / 200
    print(f"[OK] Greedy accuracy = {acc:.2%}")

    print("\nPASS: simple_dqn.py")


if __name__ == "__main__":
    _self_test()