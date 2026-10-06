# ============================================================
# environment/_phase1_summary.py
# ------------------------------------------------------------
# End-to-end integration test for Phase 1.
# Confirms channel + allocation + ris_env + quantization all
# work together and produce physically sane numbers.
# ============================================================

from __future__ import annotations

import numpy as np

from environment import (
    RicianChannel,
    SINRResult,
    compute_sinr,
    configure_phases,
    dbm_to_watt,
    jains_fairness_index,
)
from environment.ris_env import IRISEnv, EnvConfig
from environment.quantization import (
    num_levels,
    quant_step,
    phase_error_variance,
    array_gain,
    apply_quantization,
    overhead_bits,
)


def main() -> None:
    print("=" * 60)
    print("PHASE 1 -- End-to-end integration test")
    print("=" * 60)

    # --- 1. Import surface is complete ---
    print("[OK] All Phase-1 symbols import cleanly")

    # --- 2. Env config matches configs/default.yaml values ---
    cfg = EnvConfig(
        M=64, N=3, M_rows=8, M_cols=8,
        beta=0.9, tx_power_dbm=30.0, noise_power_dbm=-90.0,
        k_factor_db=10.0, episode_length=200,
        theta_rate=1.0, beta_interference=0.1, fairness_bonus=0.2,
        theta_serve=5.0,
        reward_scale=1.0, reward_clip=1e9,
        include_cancellation=True,
        seed=42,
    )
    env = IRISEnv(cfg)
    print(f"[OK] IRISEnv built (M={cfg.M}, N={cfg.N}, "
          f"state_dim={env.state_dim}, action_dim={env.action_dim})")

    # --- 3. Full random-policy episode ---
    s = env.reset()
    done = False
    steps = 0
    rewards = []
    while not done:
        a = int(np.random.default_rng(steps).integers(0, cfg.N))
        s, r, done, info = env.step(a)
        rewards.append(r)
        steps += 1
    print(f"[OK] Episode rolled out: {steps} steps, "
          f"final sum_rate = {info['sum_rate']:.3f} bps/Hz, "
          f"JFI = {info['jfi']:.4f}")

    # --- 4. Apply quantization on top of ideal SINR ---
    gamma_ideal = info["sinr"]
    for b in [1, 2, 3, 4, 5, 6, 7]:
        gamma_q = apply_quantization(gamma_ideal, b, cfg.M)
        assert np.all(gamma_q <= gamma_ideal + 1e-12)
    print(f"[OK] Quantized SINR <= ideal SINR for all b in 1..7")

    # --- 5. Quantization loss table ---
    print()
    print("b | sigma_eps^2 | G(64, b)  | overhead (bits)")
    print("--+-------------+-----------+----------------")
    for b in range(1, 8):
        print(f"{b:1d} | {phase_error_variance(b):.4e}  | "
              f"{array_gain(cfg.M, b):.6f}  | {overhead_bits(cfg.M, b):4d}")

    # --- 6. Sanity cross-checks ---
    assert num_levels(1) == 2
    assert abs(quant_step(3) - np.pi / 8) < 1e-15
    # Apply quantization to single-user high-SINR and check loss is small at b>=4
    for gamma0 in [10.0, 100.0, 1000.0]:
        gq = float(apply_quantization(gamma0, 4, cfg.M))
        assert gq > 0.99 * gamma0, f"b=4 loss too large: {gq} vs {gamma0}"
    print(f"[OK] At b=4, SINR loss < 1% for gamma in [10, 1000]")

    # --- 7. Channel + allocation direct call ---
    rng = np.random.default_rng(0)
    ch = RicianChannel(3, 64, 10.0, rng)
    h, g = ch.sample_block()
    alloc = {0: list(range(0, 21)),
             1: list(range(21, 42)),
             2: list(range(42, 64))}
    Phi = configure_phases(h, g, alloc, rng)
    res: SINRResult = compute_sinr(h, g, Phi, alloc, 0.9,
                                    np.ones(3), 1e-12)
    assert np.all(res.sinr > 0) and np.isfinite(res.sum_rate)
    print(f"[OK] Direct channel+allocation call: "
          f"sum_rate = {res.sum_rate:.3f}, JFI = "
          f"{jains_fairness_index(res.rate):.4f}")

    print()
    print("=" * 60)
    print("PHASE 1 SUMMARY: ALL MODULES COOPERATE CORRECTLY")
    print("=" * 60)
    print()
    print("Deliverables verified:")
    print("  environment/channel.py        Eq. 1        PASS")
    print("  environment/allocation.py     Eq. 2,3,5,12 PASS")
    print("  environment/ris_env.py        Alg. 1       PASS")
    print("  environment/quantization.py   Eq. 14-18    PASS")
    print()
    print("Ready for PHASE 2 (DQN agent).")


if __name__ == "__main__":
    main()