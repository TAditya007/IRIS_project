# ============================================================
# environment/allocation.py
# ------------------------------------------------------------
# RIS phase configuration, SINR, sum-rate, and fairness.
#
# Paper references:
#   Eq. (1):  received signal model (Parts I-IV)
#   Eq. (2):  SINR at user i:
#               gamma_i = beta^2 * |sum_{j in A_i} h_ij e^{j dphi_ij} g_ji sqrt(P_i)|^2
#                         / ( beta^2 * sum_{k!=i} |sum_{l in A_k} h_kl e^{j dphi_kl} g_li sqrt(P_k)|^2 + sigma_w^2 )
#   Eq. (3):  R = sum_i log2(1 + gamma_i)
#   Eq. (5):  reformulated received signal with L_i elements per user
#   Eq. (12): JFI = (sum_n R_n)^2 / (N * sum_n R_n^2)
#
# Conventions:
#   - h[i, k]   : uplink   UT_i -> element k
#   - g[i, k]   : downlink element k -> UR_i
#   - Phi[i, k] : phase applied by element k for user i
#   - allocation : dict {i: [element indices]}  (disjoint sets)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ============================================================
# Unit helpers
# ============================================================
def dbm_to_watt(p_dbm: float) -> float:
    """
    Convert dBm to Watts.

        P[W] = 10^((P[dBm] - 30) / 10)

    Example: 30 dBm -> 1 W, -90 dBm -> 1e-12 W.
    """
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


# ============================================================
# Phase configuration (Eq. 5, Part I)
# ============================================================
def configure_phases(
    h: np.ndarray,
    g: np.ndarray,
    allocation: Dict[int, List[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build the (N, M) phase matrix Phi.

    For k in allocation[i]:
        Phi[i, k] = -angle(h[i,k] * g[i,k])   -> coherent alignment
    For k not in allocation[i]:
        Phi[i, k] = Uniform(0, 2*pi)          -> random scatter (Part II)

    Parameters
    ----------
    h, g : (N, M) complex
    allocation : dict {i: [indices]}
    rng : np.random.Generator

    Returns
    -------
    Phi : (N, M) float64, values in [0, 2*pi)
    """
    n_users, n_elements = h.shape
    Phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_users, n_elements))

    for i, elems in allocation.items():
        if not elems:
            continue
        idx = np.asarray(elems, dtype=int)
        casc = h[i, idx] * g[i, idx]
        Phi[i, idx] = (-np.angle(casc)) % (2.0 * np.pi)

    return Phi


# ============================================================
# SINR / rate container
# ============================================================
@dataclass
class SINRResult:
    sinr: np.ndarray          # (N,) linear SINR per user
    rate: np.ndarray          # (N,) log2(1 + SINR) per user (Eq. 3)
    sum_rate: float           # sum_i rate_i
    signal_power: np.ndarray  # (N,) |Part I|^2 per user
    interference: np.ndarray  # (N,) |Part III|^2 per user
    noise_power: float        # sigma_w^2 (Watts)


# ============================================================
# SINR computation (Eq. 2 / Eq. 5)
# ============================================================
def compute_sinr(
    h: np.ndarray,
    g: np.ndarray,
    Phi: np.ndarray,
    allocation: Dict[int, List[int]],
    beta: float,
    tx_power_w: np.ndarray,
    noise_power_w: float,
) -> SINRResult:
    """
    Compute per-user SINR (Eq. 2) and sum-rate (Eq. 3).

    For user i:
        signal_i  = |sum_{k in A_i} h_ik e^{j Phi_ik} g_ki sqrt(P_i)|^2
        interf_i  = sum_{k != i} |sum_{l in A_k} h_kl e^{j Phi_kl} g_li sqrt(P_k)|^2
        gamma_i   = beta^2 * signal_i / (beta^2 * interf_i + sigma_w^2)
    """
    n_users, n_elements = h.shape
    signal_power = np.zeros(n_users, dtype=float)
    interference = np.zeros(n_users, dtype=float)

    # Pre-rotate cascaded channel on served elements only
    casc_served = np.zeros((n_users, n_elements), dtype=complex)
    for i, elems in allocation.items():
        if not elems:
            continue
        idx = np.asarray(elems, dtype=int)
        casc_served[i, idx] = (h[i, idx] * np.exp(1j * Phi[i, idx]) * g[i, idx])

    # ---- Signal (Part I of Eq. 5) ----
    for i in range(n_users):
        sig_sum = np.sqrt(tx_power_w[i]) * np.sum(casc_served[i])
        signal_power[i] = float(np.abs(sig_sum) ** 2)

    # ---- Interference (Part III of Eq. 5) ----
    for i in range(n_users):
        interf_acc = 0.0 + 0.0j
        for k in range(n_users):
            if k == i:
                continue
            elems = allocation.get(k, [])
            if not elems:
                continue
            idx = np.asarray(elems, dtype=int)
            contribution = (h[k, idx]
                            * np.exp(1j * Phi[k, idx])
                            * g[i, idx]
                            * np.sqrt(tx_power_w[k]))
            interf_acc += np.sum(contribution)
        interference[i] = float(np.abs(interf_acc) ** 2)

    # ---- SINR (Eq. 2) ----
    beta2 = beta ** 2
    num = beta2 * signal_power
    den = beta2 * interference + noise_power_w
    sinr = num / np.maximum(den, 1e-30)

    # ---- Rate (Eq. 3) ----
    rate = np.log2(1.0 + sinr)
    sum_rate = float(np.sum(rate))

    return SINRResult(
        sinr=sinr,
        rate=rate,
        sum_rate=sum_rate,
        signal_power=signal_power,
        interference=interference,
        noise_power=noise_power_w,
    )


# ============================================================
# Jain's Fairness Index (Eq. 12)
# ============================================================
def jains_fairness_index(rates: np.ndarray) -> float:
    """
    J(R) = (sum_n R_n)^2 / ( N * sum_n R_n^2 )     Eq. (12)

    Range: 1/N (max unfair, one user gets everything) to 1 (perfectly fair).
    """
    rates = np.asarray(rates, dtype=float)
    n = rates.size
    if n == 0:
        return 0.0
    num = float(np.sum(rates)) ** 2
    den = float(n) * float(np.sum(rates ** 2))
    return num / den if den > 0 else 0.0


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("allocation.py self-test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N, M = 3, 64

    # --- 1. Phase alignment is exact on served elements ---
    h = (rng.normal(0, np.sqrt(0.5), (N, M))
         + 1j * rng.normal(0, np.sqrt(0.5), (N, M)))
    g = (rng.normal(0, np.sqrt(0.5), (N, M))
         + 1j * rng.normal(0, np.sqrt(0.5), (N, M)))

    allocation = {
        0: list(range(0, 21)),
        1: list(range(21, 42)),
        2: list(range(42, 64)),
    }
    Phi = configure_phases(h, g, allocation, rng)

    assert Phi.shape == (N, M), f"Phi shape {Phi.shape}"
    assert np.all(Phi >= 0.0) and np.all(Phi < 2.0 * np.pi), \
        f"Phi out of [0, 2pi): [{Phi.min():.3f}, {Phi.max():.3f}]"
    print(f"[OK] Phi.shape = {Phi.shape}, range "
          f"[{Phi.min():.3f}, {Phi.max():.3f}]")

    i, k = 0, allocation[0][0]
    casc = h[i, k] * g[i, k]
    rotated = casc * np.exp(1j * Phi[i, k])
    assert abs(rotated.imag) < 1e-9, f"phase alignment not exact: {rotated}"
    assert rotated.real > 0, "rotated phasor not positive"
    print(f"[OK] Coherent phase alignment exact on served element "
          f"(rotated = {rotated.real:.4f}+{rotated.imag:.2e}j)")

    # --- 2. SINR computation ---
    beta = 0.9
    P_w = dbm_to_watt(30.0) * np.ones(N)
    sigma_w2 = dbm_to_watt(-90.0)
    print(f"[INFO] P_i = {P_w[0] * 1000.0:.2f} mW, "
          f"sigma_w^2 = {sigma_w2:.3e} W")

    res = compute_sinr(h, g, Phi, allocation, beta, P_w, sigma_w2)

    assert res.sinr.shape == (N,), f"sinr shape {res.sinr.shape}"
    assert np.all(res.sinr > 0), f"sinr not positive: {res.sinr}"
    assert np.all(np.isfinite(res.sinr)), f"sinr not finite: {res.sinr}"
    assert np.all(np.isfinite(res.rate)), f"rate not finite: {res.rate}"
    print(f"[OK] SINR (linear)  = {np.array2string(res.sinr, precision=4)}")
    print(f"[OK] Rate (bps/Hz)  = {np.array2string(res.rate, precision=4)}")
    print(f"[OK] Sum-rate       = {res.sum_rate:.4f} bps/Hz")

    # --- 3. Rate = log2(1+SINR) exactly ---
    assert np.allclose(res.rate, np.log2(1.0 + res.sinr), atol=1e-12), \
        "rate != log2(1+sinr)"
    print("[OK] Rate == log2(1 + SINR) exactly")

    # --- 4. Interference on idle users (paper's physical model) ---
    # Section I: "any subgroup of RIS elements will reflect everything it
    # receives, not just the intended user's signal."
    single_alloc = {0: list(range(M)), 1: [], 2: []}
    Phi_s = configure_phases(h, g, single_alloc, rng)
    res_s = compute_sinr(h, g, Phi_s, single_alloc, beta, P_w, sigma_w2)

    assert res_s.interference[0] == 0.0, \
        f"serving user should see zero interference: {res_s.interference[0]}"
    assert np.all(res_s.interference[[1, 2]] > 0.0), \
        f"idle users should see interference: {res_s.interference}"

    expected = beta ** 2 * M * P_w[0]
    assert np.all(res_s.interference[[1, 2]] < 5.0 * expected), \
        f"interference too large: {res_s.interference} vs expected ~{expected:.2f}"

    assert res_s.rate[0] > 0, "single-user rate should be positive"
    print(f"[OK] Single-user: interf(served) = 0, "
          f"interf(idle) = {np.array2string(res_s.interference[1:], precision=2)}, "
          f"rate(user0) = {res_s.rate[0]:.4f}")

    interf_at_L = []
    for m in [4, 16, 64]:
        a = {0: list(range(m)), 1: [], 2: []}
        P_m = configure_phases(h, g, a, rng)
        r_m = compute_sinr(h, g, P_m, a, beta, P_w, sigma_w2)
        interf_at_L.append(r_m.interference[1])
    ratio = interf_at_L[-1] / max(interf_at_L[0], 1e-12)
    assert ratio > 3.0, f"interference not scaling with L: {interf_at_L}"
    print(f"[OK] Idle-user interference scales with L: "
          f"{np.array2string(np.asarray(interf_at_L), precision=2)} "
          f"(ratio 64/4 elems = {ratio:.2f})")

    # --- 5. Coherent combining gain ---
    sinr_partial = []
    for m in [4, 8, 16, 32, 64]:
        alloc = {0: list(range(m)), 1: [], 2: []}
        Phi_m = configure_phases(h, g, alloc, rng)
        r = compute_sinr(h, g, Phi_m, alloc, beta, P_w, sigma_w2)
        sinr_partial.append(r.sinr[0])
    sinr_partial = np.array(sinr_partial)
    assert np.all(np.diff(sinr_partial) > 0), \
        f"SINR not increasing with M: {sinr_partial}"
    print(f"[OK] SINR grows with M: "
          f"{np.array2string(sinr_partial, precision=2)}")

    # --- 6. JFI checks (theoretical bounds only) ---
    jfi = jains_fairness_index(res.rate)
    assert 0.0 < jfi <= 1.0
    print(f"[OK] JFI (from real rates)  = {jfi:.4f}")

    assert abs(jains_fairness_index(np.array([1., 1., 1.])) - 1.0) < 1e-12
    print("[OK] JFI(equal rates)      = 1.0000")

    jfi_bad = jains_fairness_index(np.array([10.0, 0.01, 0.01]))
    assert jfi_bad < 0.5, f"JFI(much-unfair) too high: {jfi_bad}"
    assert jfi_bad > 1.0 / 3.0 - 1e-6, \
        f"JFI can't go below 1/N=0.3333 here: {jfi_bad}"
    print(f"[OK] JFI(unequal rates)     = {jfi_bad:.4f}  "
          f"(bounds: 1/3 <= JFI < 0.5)")

    jfi_extreme = jains_fairness_index(np.array([1.0, 0.0, 0.0]))
    assert abs(jfi_extreme - 1.0 / 3.0) < 1e-9, \
        f"JFI(one-hot) = {jfi_extreme:.6f}, expected 1/3"
    print(f"[OK] JFI(one-hot [1,0,0])   = {jfi_extreme:.4f}  "
          f"(theoretical 1/N = 0.3333)")

    # --- 7. Determinism of phase config given rng ---
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    P1 = configure_phases(h, g, allocation, r1)
    P2 = configure_phases(h, g, allocation, r2)
    assert np.allclose(P1, P2), "configure_phases not reproducible"
    print("[OK] configure_phases reproducible under fixed rng")

    print("\nPASS: allocation.py")


if __name__ == "__main__":
    _self_test()
    