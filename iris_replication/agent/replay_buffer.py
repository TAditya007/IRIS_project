# ============================================================
# agent/replay_buffer.py
# ------------------------------------------------------------
# Fixed-capacity, FIFO circular experience replay buffer.
#
# Paper reference:
#   Algorithm 1 uses a replay buffer D to break temporal
#   correlation and to train the DNN policy (Section III,
#   "Policy Update", Eq. 9):
#       L(theta) = E_{(s,a,r,s')~D} [ (y - Q_theta(s,a))^2 ]
#
# Design:
#   - Backing store is NumPy (CPU-resident) for memory efficiency
#   - FIFO: when full, oldest transitions are overwritten
#   - Uniform random sampling with replacement
#   - All randomness flows through one np.random.Generator (Rule R5)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================
# Batch container
# ============================================================
@dataclass
class Batch:
    """A sampled minibatch of transitions."""
    states: np.ndarray       # (B, state_dim)   float32
    actions: np.ndarray      # (B,)             int64
    rewards: np.ndarray      # (B,)             float32
    next_states: np.ndarray  # (B, state_dim)   float32
    dones: np.ndarray        # (B,)             float32  (0.0 or 1.0)


# ============================================================
# Replay buffer
# ============================================================
class ReplayBuffer:
    """
    FIFO replay buffer with preallocated storage.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    state_dim : int
        Dimensionality of the state vector.
    seed : int
        Seed for the internal RNG (Rule R5).
    """

    def __init__(self, capacity: int, state_dim: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {state_dim}")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        # Preallocated storage
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,), dtype=np.int64)
        self.rewards     = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=np.float32)

        # Pointers
        self.write_idx = 0    # next slot to write
        self.size = 0         # number of valid transitions

        # Single RNG for uniform sampling
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Core API
    # ---------------------------------------------------------
    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition. Overwrites oldest when full (FIFO).
        """
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        s_next = np.asarray(s_next, dtype=np.float32).reshape(-1)
        if s.shape[0] != self.state_dim:
            raise ValueError(f"state dim mismatch: {s.shape[0]} != {self.state_dim}")
        if s_next.shape[0] != self.state_dim:
            raise ValueError(f"next_state dim mismatch: {s_next.shape[0]} != {self.state_dim}")

        i = self.write_idx
        self.states[i] = s
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_states[i] = s_next
        self.dones[i] = 1.0 if done else 0.0

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a minibatch uniformly with replacement from valid slots.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self.size < batch_size:
            raise ValueError(
                f"not enough samples: size={self.size} < batch_size={batch_size}"
            )

        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            states=self.states[idx].copy(),
            actions=self.actions[idx].copy(),
            rewards=self.rewards[idx].copy(),
            next_states=self.next_states[idx].copy(),
            dones=self.dones[idx].copy(),
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    # ---------------------------------------------------------
    def clear(self) -> None:
        """Reset the buffer without reallocating."""
        self.write_idx = 0
        self.size = 0


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("replay_buffer.py self-test")
    print("=" * 60)

    STATE_DIM = 10
    CAP = 8

    buf = ReplayBuffer(capacity=CAP, state_dim=STATE_DIM, seed=42)

    # ---- 1. Starts empty ----
    assert len(buf) == 0, f"initial size {len(buf)}"
    assert buf.write_idx == 0
    print(f"[OK] Fresh buffer: len=0, capacity={CAP}")

    # ---- 2. Push fewer than capacity ----
    for i in range(5):
        s = np.full(STATE_DIM, i, dtype=np.float32)
        s2 = np.full(STATE_DIM, i + 100, dtype=np.float32)
        buf.push(s, a=i % 3, r=float(i), s_next=s2, done=False)
    assert len(buf) == 5, f"size after 5 pushes: {len(buf)}"
    assert buf.write_idx == 5
    print(f"[OK] After 5 pushes: len=5, write_idx=5")

    # ---- 3. Values are stored correctly ----
    assert buf.states[3, 0] == 3.0
    assert buf.actions[3] == 0
    assert buf.rewards[3] == 3.0
    assert buf.next_states[3, 0] == 103.0
    assert buf.dones[3] == 0.0
    print(f"[OK] Stored values correct at slot 3 "
          f"(s[0]={buf.states[3, 0]}, a={buf.actions[3]}, "
          f"r={buf.rewards[3]}, s'[0]={buf.next_states[3, 0]})")

    # ---- 4. Sample minibatch ----
    b = buf.sample(3)
    assert b.states.shape == (3, STATE_DIM)
    assert b.actions.shape == (3,)
    assert b.rewards.shape == (3,)
    assert b.next_states.shape == (3, STATE_DIM)
    assert b.dones.shape == (3,)
    assert b.states.dtype == np.float32
    assert b.actions.dtype == np.int64
    print(f"[OK] sample(3) -> shapes "
          f"states={b.states.shape}, actions={b.actions.shape}, "
          f"next_states={b.next_states.shape}, dones={b.dones.shape}")

    # ---- 5. Cannot sample more than stored ----
    try:
        buf.sample(6)
        raise AssertionError("expected ValueError for over-sample")
    except ValueError as e:
        print(f"[OK] Sampling more than stored raises: {type(e).__name__}")

    # ---- 6. FIFO eviction: push past capacity ----
    for i in range(5, 15):     # push 10 more, total 15 pushes, cap 8
        s = np.full(STATE_DIM, i, dtype=np.float32)
        s2 = np.full(STATE_DIM, i + 100, dtype=np.float32)
        buf.push(s, a=i % 3, r=float(i), s_next=s2, done=False)

    assert len(buf) == CAP, f"size capped at {len(buf)} (expected {CAP})"
    # After 15 pushes into cap=8, newest slot is at write_idx=(15 % 8)=7
    # and oldest surviving entry is the one pushed at step 7 (value 7).
    # Let's check what's in slot 0:
    #   Push sequence: values 0..14 -> slots 0..7 for 0..7, then overwrite
    #   Overwritten: 8->slot0, 9->slot1, ..., 14->slot6
    #   Slot 0 = 8, slot 1 = 9, ..., slot 6 = 14, slot 7 = 7 (oldest survivor)
    assert buf.states[0, 0] == 8.0,  f"slot 0 = {buf.states[0,0]} (expected 8)"
    assert buf.states[1, 0] == 9.0,  f"slot 1 = {buf.states[1,0]} (expected 9)"
    assert buf.states[6, 0] == 14.0, f"slot 6 = {buf.states[6,0]} (expected 14)"
    assert buf.states[7, 0] == 7.0,  f"slot 7 = {buf.states[7,0]} (expected 7)"
    print(f"[OK] FIFO eviction works: newest at write_idx={buf.write_idx}, "
          f"slot[0]={buf.states[0,0]}, slot[7]={buf.states[7,0]}")

    # ---- 7. Samples only come from valid region ----
    b = buf.sample(8)
    # Since cap=8 and full, all samples should be from indices 0..7
    unique_actions = set(b.actions.tolist())
    assert unique_actions.issubset({0, 1, 2}), f"actions {unique_actions}"
    print(f"[OK] Sampled actions from valid set: {sorted(unique_actions)}")

    # ---- 8. Reproducibility with seed ----
    buf2 = ReplayBuffer(capacity=CAP, state_dim=STATE_DIM, seed=42)
    for i in range(5):
        s = np.full(STATE_DIM, i, dtype=np.float32)
        buf2.push(s, a=i % 3, r=float(i),
                  s_next=np.full(STATE_DIM, i + 100, dtype=np.float32),
                  done=False)
    b1 = buf2.sample(3)
    # Fresh buffer with same seed should give identical sample order
    buf3 = ReplayBuffer(capacity=CAP, state_dim=STATE_DIM, seed=42)
    for i in range(5):
        s = np.full(STATE_DIM, i, dtype=np.float32)
        buf3.push(s, a=i % 3, r=float(i),
                  s_next=np.full(STATE_DIM, i + 100, dtype=np.float32),
                  done=False)
    b2 = buf3.sample(3)
    assert np.allclose(b1.states, b2.states), "sampling not reproducible"
    assert np.array_equal(b1.actions, b2.actions)
    print("[OK] Sampling reproducible under identical seed")

    # ---- 9. clear() works ----
    buf.clear()
    assert len(buf) == 0
    assert buf.write_idx == 0
    print("[OK] clear() resets size to 0")

    # ---- 10. done flag stored correctly ----
    buf = ReplayBuffer(capacity=4, state_dim=2, seed=0)
    buf.push(np.zeros(2, dtype=np.float32), 0, 0.0, np.zeros(2, dtype=np.float32), done=True)
    assert buf.dones[0] == 1.0
    print("[OK] done=True stored as 1.0")

    print("\nPASS: replay_buffer.py")


if __name__ == "__main__":
    _self_test()