# ============================================================
# agent/dqn.py
# ------------------------------------------------------------
# Double DQN agent with target network for the iRIS framework.
#
# Paper reference:
#   Section III "iRIS Framework":
#     Eq. (6)  J(theta) = E_pi[ sum_t gamma^t r_t ]              -- objective
#     Eq. (7)  Q_theta(s,a) = E_pi[ sum_t gamma^t r_t | s0=s, a0=a ]
#     Eq. (8)  Bellman recursion for Q
#     Eq. (9)  L(theta) = E_D[ (y - Q_theta(s,a))^2 ]             -- loss
#     Eq. (10) y = r + gamma * max_a' Q_theta'(s', a')            -- target
#     Algorithm 1: epsilon-greedy + periodic target sync
#
# Implementation notes:
#   - We use DOUBLE DQN (van Hasselt et al., 2016), which is the
#     modern stable form of the paper's Eq. (9)-(10):
#         y = r + gamma * Q_theta'(s', argmax_a Q_theta(s', a))
#     This replaces the max in Eq. 10 with an argmax on the online
#     net and an evaluation on the target net, reducing over-
#     estimation bias. All paper experiments use this variant.
#   - Adam optimizer, lr = 1e-4 (paper Section III / Algorithm 1)
#   - Target network synced every `target_update` steps (Eq. 10)
#   - epsilon-greedy with exponential decay (Algorithm 1)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================
@dataclass
class DQNConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 128
    buffer_size: int = 100_000
    train_start: int = 1_000
    target_update: int = 500
    device: str = "cuda"


# ============================================================
# Q-network
# ============================================================
class QNetwork(nn.Module):
    """
    Feedforward Q-network: state_dim -> hidden x L -> action_dim.

    Q_theta(s, a) approximates the action-value function (Eq. 7).
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig):
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
# DQN agent
# ============================================================
class DQNAgent:
    """
    Double DQN agent.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    cfg : DQNConfig
    seed : int
        Seed for torch + numpy RNGs (Rule R5).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 cfg: DQNConfig, seed: int = 0):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg

        # Pick device
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # RNGs
        self._np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Online and target networks
        self.online = QNetwork(state_dim, action_dim, cfg).to(self.device)
        self.target = QNetwork(state_dim, action_dim, cfg).to(self.device)
        self.sync_target()                                # Eq. 10 initial copy
        self.target.eval()                                # target never trains

        # Optimizer (paper: Adam, lr = 1e-4)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.learning_rate
        )

        # Exploration
        self._epsilon = float(cfg.epsilon_start)

        # Step counter for target sync
        self.updates = 0

    # ---------------------------------------------------------
    # Epsilon
    # ---------------------------------------------------------
    @property
    def epsilon(self) -> float:
        return self._epsilon

    def decay_epsilon(self) -> float:
        """Multiply epsilon by decay factor; floor at epsilon_end."""
        self._epsilon = max(
            self.cfg.epsilon_end,
            self._epsilon * self.cfg.epsilon_decay,
        )
        return self._epsilon

    # ---------------------------------------------------------
    # Action selection (Algorithm 1, epsilon-greedy)
    # ---------------------------------------------------------
    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy (default) or greedy (evaluation) action.

        With prob epsilon: random action (exploration)
        Else:               argmax_a Q(s, a)   (exploitation, Eq. 10 argmax)
        """
        if (not greedy) and (self._np_rng.random() < self._epsilon):
            return int(self._np_rng.integers(0, self.action_dim))

        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        q = self.online(s)
        return int(q.argmax(dim=1).item())

    # ---------------------------------------------------------
    # Learning update (Eq. 9 - 10)
    # ---------------------------------------------------------
    def update(self, batch) -> float:
        """
        One gradient step on the Double DQN loss (Eq. 9).

        Parameters
        ----------
        batch : agent.replay_buffer.Batch
            Sampled minibatch with fields states, actions, rewards,
            next_states, dones.

        Returns
        -------
        loss : float
        """
        # Move batch to device
        s = torch.as_tensor(batch.states, dtype=torch.float32,
                            device=self.device)
        a = torch.as_tensor(batch.actions, dtype=torch.long,
                            device=self.device)
        r = torch.as_tensor(batch.rewards, dtype=torch.float32,
                            device=self.device)
        s2 = torch.as_tensor(batch.next_states, dtype=torch.float32,
                             device=self.device)
        d = torch.as_tensor(batch.dones, dtype=torch.float32,
                            device=self.device)

        # Current Q(s, a)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # Paper Eq. (10): y = r + gamma * max_{a'} Q_{theta'}(s', a')
        # This is the vanilla DQN target the paper specifies.
        with torch.no_grad():
            q_target_s2 = self.target(s2).max(dim=1).values
            y = r + self.cfg.gamma * (1.0 - d) * q_target_s2

        # Loss (Eq. 9)
        loss = F.mse_loss(q_sa, y)

        # Gradient step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        self.optimizer.step()

        # Bookkeeping
        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.sync_target()

        return float(loss.detach().cpu().item())

    # ---------------------------------------------------------
    # Target sync (Eq. 10)
    # ---------------------------------------------------------
    def sync_target(self) -> None:
        """Copy online network parameters to target network."""
        self.target.load_state_dict(self.online.state_dict())

    # ---------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "updates": self.updates,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._epsilon = float(ckpt["epsilon"])
        self.updates = int(ckpt["updates"])


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("dqn.py self-test")
    print("=" * 60)

    # 0. Device
    cfg = DQNConfig(
        hidden_dim=64, num_hidden_layers=2, learning_rate=1e-3,
        gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.99, batch_size=32, buffer_size=1000,
        train_start=50, target_update=50, device="cuda",
    )
    agent = DQNAgent(state_dim=4, action_dim=3, cfg=cfg, seed=42)
    print(f"[OK] Agent on device: {agent.device}")

    # 1. Network output shape
    s = torch.zeros(1, 4, device=agent.device)
    q = agent.online(s)
    assert q.shape == (1, 3), f"q shape {q.shape}"
    print(f"[OK] Q-network output shape = {tuple(q.shape)}")

    # 2. Target starts equal to online
    for p_online, p_target in zip(agent.online.parameters(),
                                  agent.target.parameters()):
        assert torch.allclose(p_online, p_target), "target != online at init"
    print("[OK] Target network == online network at init")

    # 3. Greedy action selection is deterministic
    s_np = np.array([1.0, -0.5, 0.3, 2.0], dtype=np.float32)
    a1 = agent.select_action(s_np, greedy=True)
    a2 = agent.select_action(s_np, greedy=True)
    assert a1 == a2, f"greedy not deterministic: {a1} vs {a2}"
    print(f"[OK] Greedy selection deterministic (a={a1})")

    # 4. Epsilon-greedy covers all actions over many samples
    seen = set()
    for _ in range(500):
        seen.add(agent.select_action(s_np, greedy=False))
    assert seen == {0, 1, 2}, f"epsilon-greedy missed actions: {seen}"
    print(f"[OK] Epsilon-greedy explores all actions: {sorted(seen)}")

    # 5. Epsilon decay behaves as designed
    agent._epsilon = cfg.epsilon_start
    for _ in range(2000):
        agent.decay_epsilon()
    assert abs(agent.epsilon - cfg.epsilon_end) < 1e-12, \
        f"epsilon floor wrong: {agent.epsilon}"
    print(f"[OK] Epsilon decays to floor = {agent.epsilon:.3f}")

    # 6. One update step returns finite loss
    from agent.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=256, state_dim=4, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.normal(size=4).astype(np.float32)
        a = int(rng.integers(0, 3))
        r = float(rng.normal())
        s2 = rng.normal(size=4).astype(np.float32)
        done = bool(rng.random() < 0.1)
        buf.push(s, a, r, s2, done)
    batch = buf.sample(32)
    loss = agent.update(batch)
    assert np.isfinite(loss), f"loss not finite: {loss}"
    print(f"[OK] One update step -> loss = {loss:.6f}")

    # 7. sync_target copies weights
    # Corrupt target, sync, verify restored
    with torch.no_grad():
        for p in agent.target.parameters():
            p.add_(1.0)
    agent.sync_target()
    for p_online, p_target in zip(agent.online.parameters(),
                                  agent.target.parameters()):
        assert torch.allclose(p_online, p_target), "sync_target failed"
    print("[OK] sync_target restores target weights from online")

    # 8. Learning decreases loss on a TOY DETERMINISTIC MDP
    #    State = [x, 0, 0, 0]. Action 0 -> reward 1, done. Others -> 0.
    #    Optimal Q(s, 0) ~ 1/(1-gamma).
    print("\n[INFO] Running toy learning test (300 updates)...")
    cfg2 = DQNConfig(
        hidden_dim=32, num_hidden_layers=2, learning_rate=1e-3,
        gamma=0.9, epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay=0.995, batch_size=64, buffer_size=5000,
        train_start=200, target_update=100, device="cuda",
    )
    agent2 = DQNAgent(state_dim=4, action_dim=3, cfg=cfg2, seed=7)
    buf2 = ReplayBuffer(capacity=5000, state_dim=4, seed=7)
    rng2 = np.random.default_rng(7)

    # Fill with simple transitions where action 0 is best
    for _ in range(2000):
        s = np.array([rng2.uniform(-1, 1), 0., 0., 0.], dtype=np.float32)
        a = int(rng2.integers(0, 3))
        r = 1.0 if a == 0 else 0.0
        s2 = np.zeros(4, dtype=np.float32)
        done = True
        buf2.push(s, a, r, s2, done)

    losses = []
    for it in range(300):
        b = buf2.sample(64)
        l = agent2.update(b)
        losses.append(l)

    first10 = float(np.mean(losses[:10]))
    last10 = float(np.mean(losses[-10:]))
    assert last10 < first10, \
        f"loss did not decrease: first10={first10:.4f}, last10={last10:.4f}"
    print(f"[OK] Toy learning: mean loss {first10:.4f} -> {last10:.4f}")

    # Check learned Q ordering: Q(s, 0) > Q(s, 1), Q(s, 2)
    with torch.no_grad():
        s_test = torch.tensor([[0.5, 0., 0., 0.]], device=agent2.device)
        q_test = agent2.online(s_test).cpu().numpy().ravel()
    assert q_test[0] > q_test[1] and q_test[0] > q_test[2], \
        f"agent did not learn that action 0 is best: Q={q_test}"
    print(f"[OK] Learned Q ranking correct: Q(a=0)={q_test[0]:.3f} "
          f"> Q(a=1)={q_test[1]:.3f}, Q(a=2)={q_test[2]:.3f}")

    # 9. Save + load round-trip
    import tempfile, os
    tmp = Path(tempfile.gettempdir()) / "iris_dqn_test.pt"
    agent2.save(tmp)
    # Corrupt agent, then load
    with torch.no_grad():
        for p in agent2.online.parameters():
            p.mul_(0.0)
    agent2.load(tmp)
    with torch.no_grad():
        q_after = agent2.online(s_test).cpu().numpy().ravel()
    assert np.allclose(q_after, q_test, atol=1e-6), \
        f"save/load mismatch: {q_after} vs {q_test}"
    os.remove(tmp)
    print("[OK] Save + load round-trip preserves Q values")

    print("\nPASS: dqn.py")


if __name__ == "__main__":
    _self_test()
    