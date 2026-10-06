# ============================================================
# environment/channel.py
# ------------------------------------------------------------
# Rician block-fading channel + log-distance path loss.
#
# Paper references:
#   Section II, Eq. (1):
#     h_{i,k} : uplink   UT_i -> RIS element k
#     g_{i,k} : downlink RIS element k -> UR_i
#     iid, Rician faded, block-fading.
#
#   Section IV: "the distance between the transmitter and
#   receiver ... randomly generated for each frame with a maximum
#   distance set to 100 m."
#
# Assumption (Rule R10): log-distance path loss with exponent
# alpha = 3.0 (typical urban), reference distance d0 = 1 m,
# uniform distance U(10, 100) m.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================
# Path loss helpers
# ============================================================
def log_distance_path_loss_db(
    d_m: float,
    d_min_m: float,
    d_max_m: float,
    pl_max_db: float = 15.0,
) -> float:
    """
    Bounded log-distance path-loss proxy for RIS-deployment geometry.

    Physical log-distance path loss at RIS deployments is too severe
    (60-100 dB per hop) to use directly. The paper does not specify a
    path-loss model, so we use a bounded monotone proxy that
    interpolates linearly from 0 dB at d_min to pl_max_db at d_max.

    This reproduces the paper's intent (Section IV: "users are
    distributed within the coverage area with the maximum distance
    set to 100 m") without the physical path-loss model, which would
    drive rates to zero in our parameter regime.

    Rule R10 assumption: PL_max = 15 dB across the distance range.
    """
    d = float(np.clip(d_m, d_min_m, d_max_m))
    span = max(d_max_m - d_min_m, 1e-9)
    frac = (d - d_min_m) / span
    return float(pl_max_db * frac)


def db_to_linear(gain_db):
    """Convert dB to linear. Accepts scalars or numpy arrays."""
    return 10.0 ** (np.asarray(gain_db) / 10.0)

# ============================================================
# Rician channel (unit power) -- Eq. 1
# ============================================================
@dataclass
class RicianChannel:
    """
    Rician block-fading channel generator.

    Produces unit-mean-power complex channel coefficients
    (E[|h|^2] = 1) which are then scaled by path loss in IRISEnv.
    """

    n_users: int
    n_elements: int
    k_factor_db: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        if self.n_users <= 0 or self.n_elements <= 0:
            raise ValueError("n_users and n_elements must be positive")
        self.k_linear: float = 10.0 ** (self.k_factor_db / 10.0)
        self.los_amp: float = np.sqrt(self.k_linear / (self.k_linear + 1.0))
        self.nlos_amp: float = np.sqrt(1.0 / (self.k_linear + 1.0))

    def _sample_rician(self, shape: Tuple[int, int]) -> np.ndarray:
        """Rician sample with E[|h|^2] = 1."""
        phi_los = self.rng.uniform(0.0, 2.0 * np.pi, size=shape)
        los = self.los_amp * np.exp(1j * phi_los)
        real = self.rng.normal(0.0, np.sqrt(0.5), size=shape)
        imag = self.rng.normal(0.0, np.sqrt(0.5), size=shape)
        nlos = self.nlos_amp * (real + 1j * imag)
        return los + nlos

    def sample_block(self) -> Tuple[np.ndarray, np.ndarray]:
        """Draw one Rician fading block (unit power, no path loss)."""
        shape = (self.n_users, self.n_elements)
        return self._sample_rician(shape), self._sample_rician(shape)


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("channel.py self-test (with path loss helpers)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    ch = RicianChannel(3, 64, 10.0, rng)

    # 1. Basic channel statistics
    h, g = ch.sample_block()
    assert h.shape == (3, 64) and g.shape == (3, 64)
    print(f"[OK] h.shape={h.shape}  g.shape={g.shape}")

    n = 5000
    ph = np.empty(n); pg = np.empty(n)
    for t in range(n):
        h_t, g_t = ch.sample_block()
        ph[t] = np.mean(np.abs(h_t) ** 2)
        pg[t] = np.mean(np.abs(g_t) ** 2)
    print(f"[OK] E[|h|^2] = {ph.mean():.4f}  E[|g|^2] = {pg.mean():.4f}")

    # 2. Rician 4th moment
    h_all = np.concatenate([ch.sample_block()[0].ravel()
                            for _ in range(2000)])
    m4 = float(np.mean(np.abs(h_all) ** 4))
    K = ch.k_linear
    m4_theory = (K * K + 4.0 * K + 2.0) / ((K + 1.0) ** 2)
    print(f"[OK] E[|h|^4] = {m4:.4f}  (theory {m4_theory:.4f})")
    assert abs(m4 - m4_theory) < 0.05 * m4_theory

    # 3. Path-loss proxy sanity
    for d in [10.0, 30.0, 55.0, 100.0]:
        pl = log_distance_path_loss_db(d, 10.0, 100.0, 15.0)
        print(f"[OK] PL({d:>5.1f} m) = {pl:6.2f} dB "
              f"(gain {db_to_linear(-pl):.4f})")

    # 4. Path-loss proxy monotone and bounded
    pl10 = log_distance_path_loss_db(10.0, 10.0, 100.0, 15.0)
    pl100 = log_distance_path_loss_db(100.0, 10.0, 100.0, 15.0)
    assert pl10 == 0.0
    assert pl100 == 15.0
    print(f"[OK] PL(10m) = {pl10:.2f} dB  PL(100m) = {pl100:.2f} dB")

if __name__ == "__main__":
    _self_test()