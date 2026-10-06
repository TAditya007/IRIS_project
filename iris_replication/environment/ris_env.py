# ============================================================
# environment/ris_env.py
# ------------------------------------------------------------
# Gym-style environment for the iRIS framework.
#
# Paper references:
#   Algorithm 1 (Policy Driven DNN-Based RIS Allocation Framework).
#   Section II: sum_i L_i <= M, remaining elements in cancellation.
#   Section IV: users distributed with distances up to 100 m.
#
# Environment adds per-episode user distances (UT-RIS and RIS-UR)
# drawn uniformly in [d_min, d_max] and applies log-distance path
# loss to h and g.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment.channel import (
    RicianChannel,
    log_distance_path_loss_db,
    db_to_linear,
)
from environment.allocation import (
    compute_sinr,
    configure_phases,
    dbm_to_watt,
    jains_fairness_index,
)


# ============================================================
# Config container
# ============================================================
@dataclass
class EnvConfig:
    """All hyperparameters needed by IRISEnv."""
    M: int
    N: int
    M_rows: int
    M_cols: int
    beta: float
    tx_power_dbm: float
    noise_power_dbm: float
    k_factor_db: float
    episode_length: int
    theta_rate: float
    beta_interference: float
    fairness_bonus: float
    reward_scale: float
    reward_clip: float
    include_cancellation: bool = True
    # --- NEW: user geometry ---
    max_distance_m: float = 100.0
    min_distance_m: float = 10.0
    carrier_freq_ghz: float = 2.4
    path_loss_alpha: float = 3.0
    theta_serve: float = 30.0
    seed: int = 42


# ============================================================
# Environment
# ============================================================
class IRISEnv:
    """
    Sequential RIS element allocation with cancellation pool and
    distance-based path loss.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.channel = RicianChannel(
            n_users=cfg.N,
            n_elements=cfg.M,
            k_factor_db=cfg.k_factor_db,
            rng=self.rng,
        )

        self.element_positions: List[Tuple[int, int]] = [
            (r, c) for r in range(cfg.M_rows) for c in range(cfg.M_cols)
        ]
        assert len(self.element_positions) == cfg.M

        self.tx_power_w = np.full(cfg.N, dbm_to_watt(cfg.tx_power_dbm))
        self.noise_power_w = dbm_to_watt(cfg.noise_power_dbm)

        self.include_cancellation = bool(
            getattr(cfg, "include_cancellation", True)
        )
        if self.include_cancellation:
            self.action_dim = cfg.N + 1
            self.state_dim = 3 * cfg.N + 2
        else:
            self.action_dim = cfg.N
            self.state_dim = 3 * cfg.N + 1

        # Per-episode user geometry (updated in reset())
        self.d_ut = np.zeros(cfg.N, dtype=float)
        self.d_ur = np.zeros(cfg.N, dtype=float)

        self.h: np.ndarray = np.zeros((cfg.N, cfg.M), dtype=complex)
        self.g: np.ndarray = np.zeros((cfg.N, cfg.M), dtype=complex)
        self.allocation: Dict[int, List[int]] = {i: [] for i in range(cfg.N)}
        self.cancellation_pool: List[int] = []
        self.next_element_idx: int = 0
        self.t: int = 0

    # ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Start a new episode (Algorithm 1)."""
        # 1. Draw Rician blocks (unit-power channels)
        h_raw, g_raw = self.channel.sample_block()

        # 2. Draw user distances uniformly in [min, max] per episode
        d_min = float(self.cfg.min_distance_m)
        d_max = float(self.cfg.max_distance_m)
        self.d_ut = self.rng.uniform(d_min, d_max, size=self.cfg.N)
        self.d_ur = self.rng.uniform(d_min, d_max, size=self.cfg.N)

        # 3. Bounded path-loss proxy on the cascaded UT-RIS-UR channel.
        # Compute one path-loss proxy per user from the UT-UR distance,
        # then split the amplitude equally across h and g so that
        # |h|^2 * |g|^2 scales by 10^(-pl_db/10).
        d_total = self.d_ut + self.d_ur
        d_min = 2.0 * float(self.cfg.min_distance_m)
        d_max = 2.0 * float(self.cfg.max_distance_m)
        pl_db = np.array([
            log_distance_path_loss_db(d, d_min, d_max, pl_max_db=15.0)
            for d in d_total
        ])
        # gain_amp^2 = 10^(-pl_db/10), and it is applied to h and to g,
        # so |h|^2 * |g|^2 *= (gain_amp^2)^2 = 10^(-2*pl_db/10).
        # To make the *cascade* lose only pl_db (not 2*pl_db), we use:
        gain_amp = db_to_linear(-pl_db / 4.0)  # 4 because amplitude on both sides
        self.h = h_raw * gain_amp[:, None]
        self.g = g_raw * gain_amp[:, None]

        # 5. Reset allocation
        self.allocation = {i: [] for i in range(self.cfg.N)}
        self.cancellation_pool = []
        self.next_element_idx = 0
        self.t = 0
        return self._get_state()

    # ---------------------------------------------------------
    def step(self, action: int):
        if not (0 <= action < self.action_dim):
            raise ValueError(
                f"action {action} out of range [0, {self.action_dim})"
            )
        if self.next_element_idx >= self.cfg.M:
            raise RuntimeError("step() called after all elements allocated")

        elem = self.next_element_idx
        is_cancel = (self.include_cancellation and action == self.cfg.N)
        if is_cancel:
            self.cancellation_pool.append(elem)
        else:
            self.allocation[action].append(elem)
        self.next_element_idx += 1
        self.t += 1

        Phi = configure_phases(self.h, self.g, self.allocation, self.rng)
        res = compute_sinr(
            self.h, self.g, Phi, self.allocation,
            beta=self.cfg.beta,
            tx_power_w=self.tx_power_w,
            noise_power_w=self.noise_power_w,
        )
        jfi = jains_fairness_index(res.rate)

        # Reward (Eq. 11 + service incentive)
        reward_raw = (
            self.cfg.theta_rate * res.sum_rate
            - self.cfg.beta_interference * float(np.sum(res.interference))
            + self.cfg.fairness_bonus * jfi
        )
        for i in range(self.cfg.N):
            L_i = len(self.allocation[i])
            reward_raw += self.cfg.theta_serve * float(np.tanh(L_i / 5.0))
        reward = float(np.clip(
            reward_raw * self.cfg.reward_scale,
            -self.cfg.reward_clip,
            +self.cfg.reward_clip,
        ))

        done = self.next_element_idx >= self.cfg.M
        info = {
            "sinr": res.sinr.copy(),
            "rate": res.rate.copy(),
            "sum_rate": res.sum_rate,
            "interference": res.interference.copy(),
            "signal_power": res.signal_power.copy(),
            "jfi": jfi,
            "allocation": {k: list(v) for k, v in self.allocation.items()},
            "cancellation_pool": list(self.cancellation_pool),
            "n_cancellation": len(self.cancellation_pool),
            "d_ut": self.d_ut.copy(),
            "d_ur": self.d_ur.copy(),
            "step": self.t,
        }
        return self._get_state(), reward, bool(done), info

    # ---------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        N = self.cfg.N
        Phi = configure_phases(self.h, self.g, self.allocation, self.rng)
        res = compute_sinr(
            self.h, self.g, Phi, self.allocation,
            beta=self.cfg.beta,
            tx_power_w=self.tx_power_w,
            noise_power_w=self.noise_power_w,
        )

        sinr_part = np.log1p(np.maximum(res.sinr, 0.0)) / 10.0
        alloc_frac = np.array(
            [len(self.allocation[i]) / self.cfg.M for i in range(N)],
            dtype=float,
        )
        mean_casc = np.zeros(N, dtype=float)
        for i in range(N):
            elems = self.allocation[i]
            if elems:
                idx = np.asarray(elems, dtype=int)
                mean_casc[i] = float(
                    np.mean(np.abs(self.h[i, idx] * self.g[i, idx]))
                )
        step_frac = np.array(
            [self.t / max(1, self.cfg.episode_length)], dtype=float
        )
        if self.include_cancellation:
            cancel_frac = np.array(
                [len(self.cancellation_pool) / self.cfg.M], dtype=float
            )
            return np.concatenate(
                [sinr_part, alloc_frac, mean_casc, cancel_frac, step_frac]
            )
        else:
            return np.concatenate(
                [sinr_part, alloc_frac, mean_casc, step_frac]
            )


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("ris_env.py self-test (with path loss)")
    print("=" * 60)

    cfg = EnvConfig(
        M=64, N=3, M_rows=8, M_cols=8,
        beta=0.9, tx_power_dbm=30.0, noise_power_dbm=-10.0,
        k_factor_db=10.0, episode_length=200,
        theta_rate=1.0, beta_interference=0.5, fairness_bonus=0.0,
        reward_scale=0.1, reward_clip=1e9,
        include_cancellation=True,
        max_distance_m=100.0, min_distance_m=10.0,
        carrier_freq_ghz=2.4, path_loss_alpha=3.0,
        seed=42,
    )
    env = IRISEnv(cfg)

    print(f"[OK] state_dim={env.state_dim}  action_dim={env.action_dim}")

    s = env.reset()
    assert s.shape == (env.state_dim,)
    print(f"[OK] reset() -> state shape {s.shape}")

    # Check distances vary per episode
    d_hist = []
    for _ in range(10):
        env.reset()
        d_hist.append(env.d_ut.copy())
    d_hist = np.array(d_hist)
    print(f"[OK] d_ut across 10 episodes (first user): "
          f"{d_hist[:, 0].round(1)}")
    assert d_hist.std() > 5.0, "distances not varying per episode"

    # Episode rollout
    rng_a = np.random.default_rng(0)
    done = False
    info: Optional[Dict[str, Any]] = None
    steps = 0
    while not done:
        a = int(rng_a.integers(0, env.action_dim))
        _, _, done, info = env.step(a)
        steps += 1
    assert info is not None, "no steps executed"
    print(f"[OK] Episode rolled out: {steps} steps")
    print(f"[OK] Final sum_rate = {info['sum_rate']:.3f}  "
          f"JFI = {info['jfi']:.4f}")

    print("\nPASS: ris_env.py")


if __name__ == "__main__":
    _self_test()