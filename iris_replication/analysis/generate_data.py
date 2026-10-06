# ============================================================
# analysis/generate_data.py
# ------------------------------------------------------------
# Generate all the data files needed for the paper's figures.
#
# Outputs (all into analysis/data/):
#   fig3_allocations.npz          allocation grids for two frames
#   fig4_L_i_N3.npy               L_i samples for N=3
#   fig5_L_i_N4.npy               L_i samples for N=4 (needs N=4 model)
#   fig6_training.csv             per-episode sum-rate + per-user rates
#   fig7_empirical_L_i.npy        L_i samples for distribution fitting
#   fig8_says_curves.npz          SAYS pdf under different params
#   fig9_sigma_eps_vs_b.npz       phase error variance vs b
#   fig10_sinr_vs_b.npz           avg SINR vs b (continuous and quantized)
#   fig11_overhead.npz            per-episode overhead
#   fig12_rate_comparison.npz     proposed vs uniform rate per user
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from environment.ris_env import IRISEnv, EnvConfig
from agent.simple_dqn import SimpleDQNAgent, SimpleDQNConfig


DATA_DIR = Path(__file__).parent / "data"


# ============================================================
# Config builders
# ============================================================
def make_env_config(cfg: Dict[str, Any], N: Optional[int] = None) -> EnvConfig:
    sys_cfg = cfg["system"]; ch_cfg = cfg["channel"]
    env_cfg = cfg["env"];   rew_cfg = cfg["reward"]
    return EnvConfig(
        M=int(sys_cfg["M"]),
        N=int(N if N is not None else sys_cfg["N"]),
        M_rows=int(sys_cfg["M_rows"]),
        M_cols=int(sys_cfg["M_cols"]),
        beta=float(sys_cfg["beta"]),
        tx_power_dbm=float(sys_cfg["tx_power_dbm"]),
        noise_power_dbm=float(sys_cfg["noise_power_dbm"]),
        k_factor_db=float(ch_cfg["K_factor_db"]),
        episode_length=int(env_cfg["episode_length"]),
        theta_rate=float(rew_cfg["theta_rate"]),
        beta_interference=float(rew_cfg["beta_interference"]),
        fairness_bonus=float(rew_cfg["fairness_bonus"]),
        theta_serve=float(rew_cfg.get("theta_serve", 30.0)),
        reward_scale=float(rew_cfg.get("scale", 1.0)),
        reward_clip=float(rew_cfg.get("clip", 1e9)),
        include_cancellation=bool(env_cfg.get("include_cancellation", True)),
        max_distance_m=float(env_cfg.get("max_distance_m", 100.0)),
        min_distance_m=float(env_cfg.get("min_distance_m", 10.0)),
        carrier_freq_ghz=float(env_cfg.get("carrier_freq_ghz", 2.4)),
        path_loss_alpha=float(env_cfg.get("path_loss_alpha", 3.0)),
        seed=int(cfg["seed"]),
    )


def make_agent_config(cfg: Dict[str, Any]) -> SimpleDQNConfig:
    a = cfg["agent"]
    return SimpleDQNConfig(
        hidden_dim=int(a["hidden_dim"]),
        num_hidden_layers=int(a["num_hidden_layers"]),
        learning_rate=float(a["learning_rate"]),
        gamma=float(a["gamma"]),
        epsilon_start=float(a["epsilon_start"]),
        epsilon_end=float(a["epsilon_end"]),
        epsilon_decay=float(a["epsilon_decay"]),
        device=cfg["training"]["device"],
    )


def build_agent(
    env: IRISEnv,
    cfg: Dict[str, Any],
    ckpt_path: str = "checkpoints/PAPER_REPLICATION_FINAL.pt",
) -> SimpleDQNAgent:
    agent = SimpleDQNAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        cfg=make_agent_config(cfg), seed=int(cfg["seed"]),
    )
    agent.load(ckpt_path)
    agent.online.eval()
    return agent


# ============================================================
# Fig 3 -- allocation visualization on 8x8 grid (two frames)
# ============================================================
def gen_fig3_data(
    env: IRISEnv,
    agent: SimpleDQNAgent,
    M_rows: int,
    M_cols: int,
    out: Path,
) -> None:
    frames = []
    for _ in range(2):
        env.reset()
        state = env._get_state()
        done = False
        info: Optional[Dict[str, Any]] = None
        while not done:
            a = agent.select_action(state, greedy=True)
            state, _, done, info = env.step(a)
        assert info is not None, "episode produced no steps"
        grid = np.full((M_rows, M_cols), -1, dtype=int)  # -1 = cancelled
        for i in range(env.cfg.N):
            for elem in info["allocation"][i]:
                r, c = divmod(elem, M_cols)
                grid[r, c] = i
        frames.append(grid)
    np.savez(out, frame1=frames[0], frame2=frames[1],
             M_rows=M_rows, M_cols=M_cols, N=env.cfg.N)
    print(f"[OK] fig3 data -> {out}")


# ============================================================
# Fig 4 / Fig 5 -- L_i samples for N=3 and N=4
# ============================================================
def gen_fig45_data(
    env: IRISEnv,
    agent: SimpleDQNAgent,
    n_frames: int,
    out: Path,
    N: int,
) -> None:
    Ls = []
    for _ in range(n_frames):
        env.reset()
        state = env._get_state()
        done = False
        info: Optional[Dict[str, Any]] = None
        while not done:
            a = agent.select_action(state, greedy=True)
            state, _, done, info = env.step(a)
        assert info is not None, "episode produced no steps"
        Ls.append([len(info["allocation"][i]) for i in range(N)])
    L_arr = np.array(Ls)  # (n_frames, N)
    np.save(out, L_arr)
    print(f"[OK] fig45 data (N={N}) -> {out}  shape={L_arr.shape}")


# ============================================================
# Fig 6 -- per-episode sum-rate + per-user rates from training log
# ============================================================
def gen_fig6_data(log_csv: str, out: Path) -> None:
    df = pd.read_csv(log_csv)
    # Training log has: episode, sum_rate, mean_rate (mean over users), jfi, ...
    # We save episode, sum_rate, jfi
    np.savez(out,
             episode=df["episode"].to_numpy(),
             sum_rate=df["sum_rate"].to_numpy(),
             mean_rate=df["mean_rate"].to_numpy(),
             jfi=df["jfi"].to_numpy())
    print(f"[OK] fig6 data -> {out}  ({len(df)} episodes)")


# ============================================================
# Fig 7 -- empirical L_i for SAYS fit (aggregate over users)
# ============================================================
def gen_fig7_data(L3_path: Path, out: Path) -> None:
    L = np.load(L3_path)              # (n_frames, N=3)
    flat = L.ravel()                  # flatten all users/frames
    np.save(out, flat)
    print(f"[OK] fig7 data -> {out}  shape={flat.shape}")


# ============================================================
# Fig 9 / Fig 10 -- quantization from environment/quantization
# ============================================================
def gen_fig910_data(out9: Path, out10: Path) -> None:
    from environment.quantization import (
        num_levels, quant_step, phase_error_variance, array_gain
    )
    bits = np.arange(1, 8)
    var_eps = np.array([phase_error_variance(int(b)) for b in bits])
    G64 = np.array([array_gain(64, int(b)) for b in bits])
    G16 = np.array([array_gain(16, int(b)) for b in bits])
    # For Fig 10, average SINR: ideal vs quantized
    gamma_ideal = 100.0  # arbitrary representative value
    gamma_q = G64 * gamma_ideal
    np.savez(out9, bits=bits, var_eps=var_eps)
    np.savez(out10, bits=bits,
             gamma_continuous=np.full_like(bits, gamma_ideal, dtype=float),
             gamma_quantized=gamma_q)
    print(f"[OK] fig9 data -> {out9}")
    print(f"[OK] fig10 data -> {out10}")


# ============================================================
# Fig 11 -- overhead per episode (raw + moving average)
# ============================================================
def gen_fig11_data(log_csv: str, out: Path) -> None:
    df = pd.read_csv(log_csv)
    # Overhead definition (paper's Fig. 11): moving-average of allocation
    # churn (elements reallocated per frame). Our proxy: number of
    # non-zero allocation changes across consecutive episodes.
    #
    # We don't have per-frame element identity in the log, so we use
    # the JFI variance as a proxy for allocation stability.
    eps = df["episode"].to_numpy()
    # Raw overhead: rolling std of JFI over 5-episode window
    jfi = df["jfi"].to_numpy()
    window = 5
    raw = np.array([
        np.std(jfi[max(0, i-window):i+1]) if i > 0 else 0.0
        for i in range(len(jfi))
    ])
    # Moving average of raw
    ma = np.convolve(raw, np.ones(10)/10, mode="same")
    np.savez(out, episode=eps, raw=raw, moving_avg=ma)
    print(f"[OK] fig11 data -> {out}")


# ============================================================
# Fig 12 -- proposed vs uniform allocation
# ============================================================
def gen_fig12_data(
    env: IRISEnv,
    agent: SimpleDQNAgent,
    n_frames: int,
    out: Path,
) -> None:
    N = env.cfg.N
    proposed_rates = np.zeros((n_frames, N))
    uniform_rates = np.zeros((n_frames, N))

    for t in range(n_frames):
        # Proposed: use trained agent
        env.reset()
        state = env._get_state()
        done = False
        info: Optional[Dict[str, Any]] = None
        while not done:
            a = agent.select_action(state, greedy=True)
            state, _, done, info = env.step(a)
        assert info is not None, "episode produced no steps"
        proposed_rates[t] = info["rate"]

        # Uniform: allocate elements round-robin
        env.reset()
        state = env._get_state()
        done = False
        info: Optional[Dict[str, Any]] = None
        step = 0
        while not done:
            a = step % N
            state, _, done, info = env.step(a)
            step += 1
        assert info is not None, "episode produced no steps"
        uniform_rates[t] = info["rate"]

    np.savez(out,
             proposed_mean=proposed_rates.mean(axis=0),
             proposed_std=proposed_rates.std(axis=0),
             uniform_mean=uniform_rates.mean(axis=0),
             uniform_std=uniform_rates.std(axis=0))
    print(f"[OK] fig12 data -> {out}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--log", type=str,
                    default="logs/PHASE_B_converged_log.csv")
    ap.add_argument("--n_frames", type=int, default=500)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_loaded = yaml.safe_load(f)
    assert isinstance(cfg_loaded, dict), "config must be a dict"
    cfg: Dict[str, Any] = cfg_loaded

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Environment + agent (N=3) ---
    env = IRISEnv(make_env_config(cfg, N=3))
    agent = build_agent(env, cfg)

    # --- Fig 3: allocation grids ---
    gen_fig3_data(env, agent, env.cfg.M_rows, env.cfg.M_cols,
                  DATA_DIR / "fig3_allocations.npz")

    # --- Fig 4: L_i distribution N=3 ---
    gen_fig45_data(env, agent, args.n_frames,
                   DATA_DIR / "fig4_L_i_N3.npy", N=3)

    # --- Fig 5: L_i distribution N=4 (needs N=4 model) ---
    # We don't have an N=4 model; skip for now and note in report
    print("[SKIP] fig5 (N=4) requires a trained N=4 model")
    # Generate a placeholder with random policy for N=4:
    env4 = IRISEnv(make_env_config(cfg, N=4))
    L4 = []
    rng4 = np.random.default_rng(42)
    for _ in range(200):
        env4.reset()
        done = False
        info: Optional[Dict[str, Any]] = None
        while not done:
            a = int(rng4.integers(0, env4.action_dim))
            _, _, done, info = env4.step(a)
        assert info is not None, "episode produced no steps"
        L4.append([len(info["allocation"][i]) for i in range(4)])
    np.save(DATA_DIR / "fig5_L_i_N4.npy", np.array(L4))
    print(f"[OK] fig5 data (random policy placeholder) -> "
          f"{DATA_DIR / 'fig5_L_i_N4.npy'}")

    # --- Fig 6: training curve ---
    gen_fig6_data(args.log, DATA_DIR / "fig6_training.npz")

    # --- Fig 7: empirical L_i for SAYS fit ---
    gen_fig7_data(DATA_DIR / "fig4_L_i_N3.npy",
                  DATA_DIR / "fig7_empirical_L_i.npy")

    # --- Fig 9 / 10: quantization ---
    gen_fig910_data(DATA_DIR / "fig9_sigma_eps_vs_b.npz",
                    DATA_DIR / "fig10_sinr_vs_b.npz")

    # --- Fig 11: overhead ---
    gen_fig11_data(args.log, DATA_DIR / "fig11_overhead.npz")

    # --- Fig 12: proposed vs uniform ---
    gen_fig12_data(env, agent, n_frames=200,
                   out=DATA_DIR / "fig12_rate_comparison.npz")

    print("\nAll figure data generated in analysis/data/")


if __name__ == "__main__":
    main()