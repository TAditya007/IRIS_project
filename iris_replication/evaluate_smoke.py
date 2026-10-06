# ============================================================
# evaluate_smoke.py
# Evaluate a checkpoint on a held-out set of channels.
# ============================================================
from __future__ import annotations

import argparse
import numpy as np
import torch
import yaml

from environment.ris_env import IRISEnv, EnvConfig
from agent.simple_dqn import SimpleDQNAgent, SimpleDQNConfig


def build_env(cfg):
    sys_cfg = cfg["system"]
    ch_cfg = cfg["channel"]
    env_cfg = cfg["env"]
    rew_cfg = cfg["reward"]
    return IRISEnv(EnvConfig(
        M=int(sys_cfg["M"]),
        N=int(sys_cfg["N"]),
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
    ))


def build_agent(env, cfg, ckpt_path):
    a = cfg["agent"]
    dqn_cfg = SimpleDQNConfig(
        hidden_dim=int(a["hidden_dim"]),
        num_hidden_layers=int(a["num_hidden_layers"]),
        learning_rate=float(a["learning_rate"]),
        gamma=float(a["gamma"]),
        epsilon_start=float(a["epsilon_start"]),
        epsilon_end=float(a["epsilon_end"]),
        epsilon_decay=float(a["epsilon_decay"]),
        device=cfg["training"]["device"],
    )
    agent = SimpleDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cfg=dqn_cfg,
        seed=int(cfg["seed"]),
    )
    agent.load(ckpt_path)
    agent.online.eval()
    return agent


def evaluate(env, agent, n_channels=200, seed=999):
    sum_rates, jfis, alloc_hist = [], [], []
    for _ in range(n_channels):
        state = env.reset()
        done = False
        info = None
        while not done:
            a = agent.select_action(state, greedy=True)
            state, _, done, info = env.step(a)
        sum_rates.append(info["sum_rate"])
        jfis.append(info["jfi"])
        alloc_hist.append([len(info["allocation"][i]) for i in range(env.cfg.N)])
    return np.array(sum_rates), np.array(jfis), np.array(alloc_hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--n_channels", type=int, default=200)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = build_env(cfg)
    agent = build_agent(env, cfg, args.ckpt)
    sr, jf, allocs = evaluate(env, agent, n_channels=args.n_channels)

    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print("=" * 60)
    print(f"N channels           : {args.n_channels}")
    print()
    print(f"Sum-rate mean        : {sr.mean():.4f} bps/Hz")
    print(f"Sum-rate std         : {sr.std():.4f}")
    print(f"Sum-rate min / max   : {sr.min():.4f} / {sr.max():.4f}")
    print()
    print(f"JFI mean             : {jf.mean():.4f}")
    print(f"JFI std              : {jf.std():.4f}")
    print(f"JFI min / max        : {jf.min():.4f} / {jf.max():.4f}")
    print()
    print(f"L_i mean per user    : {allocs.mean(axis=0)}")
    print(f"L_i std  per user    : {allocs.std(axis=0)}")
    print(f"L_i overall mean     : {allocs.mean():.2f}")
    print(f"L_i overall std      : {allocs.std():.2f}")
    print()
    print("L_i histogram (per-user element counts, 10-element bins):")
    M = env.cfg.M
    for lo in range(0, M + 1, 10):
        hi = min(lo + 10, M + 1)
        mask = (allocs >= lo) & (allocs < hi)
        count = int(mask.sum())
        total = int(allocs.size)
        pct = 100.0 * count / max(total, 1)
        bar = "#" * int(pct / 2)
        print(f"  [{lo:3d}-{hi:3d}) {count:5d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()