# ============================================================
# environment/quantization.py
# ------------------------------------------------------------
# Phase quantization for RIS elements (Section IV-A, Eq. 14-18).
#
# Paper reference:
#   Eq. (14): L = 2^b              -- number of quantization levels
#   Eq. (15): Delta = pi / 2^b     -- quantization step
#   Eq. (16): eps in [-Delta/2, Delta/2]  -- phase error bound
#             var(eps) = Delta^2 / 12 (uniform quantization error)
#   Eq. (17): G(b) = 1/M + (1 - 1/M) |Phi|^2  for small M
#             G(b) = |Phi|^2                     for large M
#             where Phi = E[e^{j eps}] = sinc(Delta/2)
#             sinc(x) = sin(x)/x
#   Eq. (18): gamma^(Q) ~= G(b) * gamma
#
# Convention:
#   - Ideal phase theta in [0, 2*pi)
#   - b-bit quantization: k*Delta, k in {0, 1, ..., L-1}
# ============================================================

from __future__ import annotations

import numpy as np


# ============================================================
# Basic quantities (Eq. 14 - 16)
# ============================================================
def num_levels(b: int) -> int:
    """
    Eq. (14): L = 2^b.

    Parameters
    ----------
    b : int, >= 1
        Number of quantization bits.

    Returns
    -------
    L : int
        Number of phase levels.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    return 1 << b          # = 2 ** b, but faster and exact


def quant_step(b: int) -> float:
    """
    Eq. (15): Delta = pi / 2^b.
    """
    return np.pi / num_levels(b)


def phase_error_variance(b: int) -> float:
    """
    Uniform quantization error variance (text after Eq. 16):

        var(eps) = Delta^2 / 12

    for eps ~ Uniform(-Delta/2, +Delta/2).
    """
    D = quant_step(b)
    return D * D / 12.0


# ============================================================
# Array gain (Eq. 17)
# ============================================================
def _phi_expectation(b: int) -> float:
    """
    Phi = E[e^{j eps}] for eps ~ Uniform(-Delta/2, +Delta/2).

    Phi = (1/Delta) * int_{-Delta/2}^{+Delta/2} e^{j eps} d eps
        = sin(Delta/2) / (Delta/2)
        = sinc(Delta / 2)          [with sinc(x) = sin(x)/x]

    Returns
    -------
    Phi : float, in [0, 1]
    """
    D = quant_step(b)
    x = D / 2.0
    if x < 1e-15:
        return 1.0                 # b -> inf -> no loss
    # np.sinc(x) = sin(pi*x) / (pi*x); we want sin(x)/x
    return float(np.sin(x) / x)


def array_gain(M: int, b: int) -> float:
    """
    Eq. (17): normalized RIS array gain under b-bit phase quantization.

        G(b) = 1/M + (1 - 1/M) * |Phi|^2      for small M
        G(b) = |Phi|^2                        for large M

    Here we return the small-M expression (which naturally approaches
    the large-M form as M -> infinity).

    Parameters
    ----------
    M : int, >= 1
        Number of RIS elements.
    b : int, >= 1
        Phase quantization bits.

    Returns
    -------
    G : float, in (0, 1]
        G = 1 means no quantization loss; G < 1 means loss.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    Phi = _phi_expectation(b)
    return 1.0 / M + (1.0 - 1.0 / M) * (Phi * Phi)


def array_gain_large_M(b: int) -> float:
    """
    Eq. (17), large-M limit: G = |Phi|^2.
    """
    Phi = _phi_expectation(b)
    return Phi * Phi


# ============================================================
# Quantized phase grid (Eq. 14-16, application)
# ============================================================
def quantize_phase(theta: np.ndarray | float, b: int) -> np.ndarray:
    """
    Round ideal phases theta in [0, 2*pi) to the nearest level
    on the b-bit grid {0, Delta, 2*Delta, ..., (L-1)*Delta}.

    Phase-error bound: |theta_q - theta| <= Delta/2 (Eq. 16).

    Parameters
    ----------
    theta : array_like or float
        Ideal phases in radians, assumed already wrapped to [0, 2*pi).
    b : int, >= 1

    Returns
    -------
    theta_q : ndarray (same shape as theta, float64)
        Quantized phases, all in [0, 2*pi).
    """
    theta = np.asarray(theta, dtype=float)
    D = quant_step(b)
    # round to nearest grid point, then wrap back into [0, 2*pi)
    k = np.round(theta / D)
    theta_q = np.mod(k * D, 2.0 * np.pi)
    return theta_q


def phase_error(theta: np.ndarray | float, b: int) -> np.ndarray:
    """
    Phase quantization error eps = theta_q - theta,
    wrapped into (-pi, +pi] to remove 2*pi ambiguity.

    Bound from Eq. (16): |eps| <= Delta/2.
    """
    theta = np.asarray(theta, dtype=float)
    theta_q = quantize_phase(theta, b)
    err = theta_q - theta
    # wrap to (-pi, pi]
    err = (err + np.pi) % (2.0 * np.pi) - np.pi
    return err


# ============================================================
# Effective SINR (Eq. 18)
# ============================================================
def apply_quantization(gamma: np.ndarray | float, b: int, M: int) -> np.ndarray:
    """
    Eq. (18): gamma^(Q) ~= G(b) * gamma.

    Coherent combining gain under b-bit quantization is scaled
    down by G(b) (Eq. 17). Interference scales identically because
    it also depends on the same coherent sum of reflected paths.

    Parameters
    ----------
    gamma : array_like or float
        Ideal (unquantized) SINR values.
    b : int, >= 1
    M : int, >= 1

    Returns
    -------
    gamma_q : ndarray (same shape as gamma, float64)
        Effective SINR under b-bit phase quantization.
    """
    gamma = np.asarray(gamma, dtype=float)
    G = array_gain(M, b)
    return G * gamma


# ============================================================
# Overhead (paper Section IV, "control signaling")
# ============================================================
def overhead_bits(M: int, b: int) -> int:
    """
    Bits needed to configure one RIS frame with M elements at b-bit
    phase resolution.

        bits_per_frame = M * b

    Used later for Fig. 11 (overhead).
    """
    if M < 1 or b < 1:
        raise ValueError(f"M and b must be >= 1, got M={M}, b={b}")
    return M * b


# ============================================================
# Self-test
# ============================================================
def _self_test() -> None:
    print("=" * 60)
    print("quantization.py self-test")
    print("=" * 60)

    # ---- 1. Eq. 14: L = 2^b ----
    for b in [1, 2, 3, 4, 5, 6, 7]:
        L = num_levels(b)
        assert L == 2 ** b, f"L({b}) = {L}, expected {2 ** b}"
    print(f"[OK] Eq.14  L = 2^b  for b in 1..7: "
          f"{[num_levels(b) for b in range(1, 8)]}")

    # ---- 2. Eq. 15: Delta = pi / 2^b ----
    for b in [1, 2, 3, 4, 5, 6, 7]:
        D = quant_step(b)
        expected = np.pi / (2 ** b)
        assert abs(D - expected) < 1e-15, f"Delta({b}) = {D} != {expected}"
    print(f"[OK] Eq.15  Delta = pi / 2^b, "
          f"Delta(3) = {quant_step(3):.6f} rad "
          f"({np.degrees(quant_step(3)):.3f} deg)")

    # ---- 3. Phase error variance: sigma_eps^2 = Delta^2 / 12 ----
    for b in [1, 2, 3, 4]:
        v = phase_error_variance(b)
        expected = quant_step(b) ** 2 / 12.0
        assert abs(v - expected) < 1e-18, f"var({b}) = {v} != {expected}"
    print(f"[OK] sigma_eps^2 = Delta^2/12: "
          f"{[f'{phase_error_variance(b):.4e}' for b in range(1, 5)]}")

    # ---- 4. Equipartition of quantization levels ----
    # Check b-bit phase quantization on 0..2*pi has |error| <= Delta/2
    rng = np.random.default_rng(42)
    for b in [1, 2, 3, 4, 5]:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=10000)
        eps = phase_error(theta, b)
        D = quant_step(b)
        assert np.all(np.abs(eps) <= D / 2.0 + 1e-12), \
            f"phase error exceeds Delta/2 at b={b}: max={np.abs(eps).max()}"
    print("[OK] Eq.16  |eps| <= Delta/2 verified for b in 1..5")

    # ---- 5. Empirical variance of eps matches Delta^2/12 ----
    theta = rng.uniform(0.0, 2.0 * np.pi, size=200000)
    for b in [2, 3, 4, 5]:
        eps = phase_error(theta, b)
        var_emp = float(np.var(eps))
        var_theory = phase_error_variance(b)
        # allow ~5% deviation for a finite sample
        assert abs(var_emp - var_theory) / var_theory < 0.05, \
            f"empirical var({b}) = {var_emp:.4e} vs theory {var_theory:.4e}"
    print("[OK] Empirical Var(eps) matches Delta^2/12 within 5%")

    # ---- 6. Eq. 17: G(b) bounded and monotone in b ----
    M = 64
    G_vals = [array_gain(M, b) for b in range(1, 8)]
    for i in range(len(G_vals) - 1):
        assert G_vals[i] < G_vals[i + 1], \
            f"G not monotone: {G_vals}"
    for G in G_vals:
        assert 0.0 < G <= 1.0, f"G out of (0,1]: {G}"
    print(f"[OK] Eq.17  G(M={M}, b) monotone in b: "
          f"{[f'{g:.4f}' for g in G_vals]}")

    # ---- 7. Eq. 17: G(b->inf) -> 1 ----
    for b in [12, 16]:
        G = array_gain(M, b)
        assert abs(G - 1.0) < 1e-3, f"G(large b={b}) = {G}, expected ~1"
    print(f"[OK] Eq.17  G(b=16, M={M}) = {array_gain(M, 16):.6f}  (~1, no loss)")

    # ---- 8. Eq. 17: large-M limit is |Phi|^2 ----
    for b in [1, 2, 3, 4]:
        G_small = array_gain(M, b)
        G_large = array_gain_large_M(b)
        assert G_small >= G_large - 1e-12, \
            f"G_small < G_large: {G_small} < {G_large}"
    print(f"[OK] Eq.17  G_small >= G_large for all b "
          f"(difference = 1/M scale)")

    # ---- 9. Eq. 18: gamma^(Q) = G * gamma exactly ----
    gammas = np.array([1.0, 5.0, 20.0])
    for b in [1, 3, 5]:
        gq = apply_quantization(gammas, b, M)
        G = array_gain(M, b)
        assert np.allclose(gq, G * gammas, rtol=1e-12)
    print("[OK] Eq.18  gamma^(Q) = G(b) * gamma exactly")

    # ---- 10. gamma^(Q) <= gamma always (quantization can only hurt) ----
    for b in [1, 2, 3, 4, 5, 6, 7]:
        gq = apply_quantization(gammas, b, M)
        assert np.all(gq <= gammas + 1e-12), \
            f"quantization improved SINR at b={b}: {gq} > {gammas}"
    print("[OK] gamma^(Q) <= gamma always (quantization is lossless only at b->inf)")

    # ---- 11. Reproduce Fig. 9 qualitative trend ----
    # Fig. 9 plots sigma_eps^2 vs b on a log scale; must be strictly decreasing.
    var_vals = [phase_error_variance(b) for b in range(1, 8)]
    for i in range(len(var_vals) - 1):
        assert var_vals[i] > var_vals[i + 1], \
            f"Var(eps) not decreasing: {var_vals}"
    print(f"[OK] Fig.9 trend (sigma_eps^2 decreasing): "
          f"{[f'{v:.4e}' for v in var_vals]}")

    # ---- 12. Reproduce Fig. 10 qualitative trend ----
    # Fig. 10: average SINR vs b; b=1 has biggest loss, saturates as b grows.
    avg_gamma_ideal = 100.0
    avg_gamma_q = [float(apply_quantization(avg_gamma_ideal, b, M))
                   for b in range(1, 8)]
    for i in range(len(avg_gamma_q) - 1):
        assert avg_gamma_q[i] < avg_gamma_q[i + 1]
    assert avg_gamma_q[-1] / avg_gamma_ideal > 0.99, \
        f"b=7 gain only {avg_gamma_q[-1] / avg_gamma_ideal:.4f}"
    print(f"[OK] Fig.10 trend (avg SINR approaching ideal): "
          f"{[f'{g:.2f}' for g in avg_gamma_q]}")

    # ---- 13. Overhead scales linearly in M and b ----
    for M_test in [16, 64, 256]:
        for b in [1, 3, 5]:
            bits = overhead_bits(M_test, b)
            assert bits == M_test * b, \
                f"overhead({M_test},{b}) = {bits} != {M_test * b}"
    print(f"[OK] Overhead = M * b bits/frame: "
          f"M=64,b=3 -> {overhead_bits(64, 3)} bits")

    print("\nPASS: quantization.py")


if __name__ == "__main__":
    _self_test()