# ============================================================
# train_alg1.py
# ------------------------------------------------------------
# Literal implementation of the paper's Algorithm 1.
#
# Differences from train.py:
#   - No replay buffer. Each env step performs one gradient
#     descent step on the current single transition.
#   - Uses SimpleDQNAgent (single network, no target copy).
#   - Held-out eval retained as evidence only (does not affect
#     training).
#
# Run:  python train_alg1.py
# ============================================================

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from environment.ris_env import IRISEnv, EnvConfig
from agent.simple_dqn import SimpleDQNAgent, SimpleDQNConfig


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_config(cfg: Dict[str, Any]) -> EnvConfig:
    sys_cfg = cfg["system"]; ch_cfg = cfg["channel"]
    env_cfg = cfg["env"];   rew_cfg = cfg["reward"]
    return EnvConfig(
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
    )


def make_simple_cfg(cfg: Dict[str, Any], force_cpu: bool) -> SimpleDQNConfig:
    a = cfg["agent"]
    device = "cpu" if force_cpu else cfg["training"]["device"]
    return SimpleDQNConfig(
        hidden_dim=int(a["hidden_dim"]),
        num_hidden_layers=int(a["num_hidden_layers"]),
        learning_rate=float(a["learning_rate"]),
        gamma=float(a["gamma"]),
        epsilon_start=float(a["epsilon_start"]),
        epsilon_end=float(a["epsilon_end"]),
        epsilon_decay=float(a["epsilon_decay"]),
        device=device,
    )


def eval_greedy(env, agent, n_channels=50):
    """Evaluate the current greedy policy on the training environment
    distribution (with path loss, as during training)."""
    sr, jf = [], []
    N = env.cfg.N
    saved_rng = env.rng
    saved_h = env.h.copy(); saved_g = env.g.copy()
    saved_alloc = {k: list(v) for k, v in env.allocation.items()}
    saved_next = env.next_element_idx; saved_t = env.t
    saved_d_ut = env.d_ut.copy(); saved_d_ur = env.d_ur.copy()

    # Do NOT reset env.rng — we want eval on the training distribution
    try:
        for _ in range(n_channels):
            state = env.reset()
            done = False
            info = None
            while not done:
                a = agent.select_action(state, greedy=True)
                state, _, done, info = env.step(a)
            sr.append(info["sum_rate"]); jf.append(info["jfi"])
    finally:
        env.rng = saved_rng
        env.h, env.g = saved_h, saved_g
        env.allocation = saved_alloc
        env.next_element_idx = saved_next; env.t = saved_t
        env.d_ut, env.d_ur = saved_d_ut, saved_d_ur
    return float(np.mean(sr)), float(np.mean(jf))


class TrainLogger:
    FIELDS = ["episode", "steps", "sum_rate", "mean_rate", "jfi",
              "total_reward", "mean_reward", "mean_loss", "epsilon",
              "n_updates", "wall_time_s", "eval_sum_rate", "eval_jfi"]
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
    def log(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


def train(cfg, env_cfg, agent_cfg, num_episodes, seed, run_dir):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = IRISEnv(env_cfg)
    agent = SimpleDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                           cfg=agent_cfg, seed=seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / "logs" / f"train_alg1_log_{ts}.csv"
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(log_path)

    save_interval = int(cfg["training"]["save_interval"])
    log_interval = int(cfg["training"]["log_interval"])

    print(f"[INFO] env: M={env_cfg.M}, N={env_cfg.N}, state_dim={env.state_dim}")
    print(f"[INFO] agent on device: {agent.device}")
    print(f"[INFO] Algorithm 1 literal: no buffer, no target net, batch=1")
    print(f"[INFO] training for {num_episodes} episodes")
    print(f"[INFO] log: {log_path}")

    t_start = time.time(); global_step = 0
    episode_stats: List[Dict[str, Any]] = []

    pbar = tqdm(range(1, num_episodes + 1), desc="train", unit="ep", ncols=110)
    for episode in pbar:
        state = env.reset()
        ep_reward = 0.0; ep_losses = []; ep_info = None; steps = 0; done = False

        while not done:
            action = agent.select_action(state, greedy=False)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward; ep_info = info; steps += 1

            loss = agent.update_one(state, action, reward, next_state, done)
            ep_losses.append(loss)

            state = next_state; global_step += 1

        agent.decay_epsilon()

        eval_R = float("nan"); eval_J = float("nan")
        if episode % log_interval == 0:
            eval_R, eval_J = eval_greedy(env, agent, n_channels=50)

        sum_rate = float(ep_info["sum_rate"]) if ep_info else 0.0
        rate_vec = ep_info["rate"] if ep_info else np.zeros(env_cfg.N)
        jfi = float(ep_info["jfi"]) if ep_info else 0.0
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        row = {
            "episode": episode, "steps": steps,
            "sum_rate": round(sum_rate, 6),
            "mean_rate": round(float(np.mean(rate_vec)), 6),
            "jfi": round(jfi, 6),
            "total_reward": round(ep_reward, 6),
            "mean_reward": round(ep_reward / max(steps, 1), 6),
            "mean_loss": round(mean_loss, 6),
            "epsilon": round(agent.epsilon, 6),
            "n_updates": agent.updates,
            "wall_time_s": round(time.time() - t_start, 2),
            "eval_sum_rate": round(eval_R, 6) if not np.isnan(eval_R) else "",
            "eval_jfi": round(eval_J, 6) if not np.isnan(eval_J) else "",
        }
        logger.log(row); episode_stats.append(row)

        pbar.set_postfix({"R": f"{sum_rate:.2f}", "JFI": f"{jfi:.3f}",
                          "eps": f"{agent.epsilon:.3f}", "loss": f"{mean_loss:.2f}"})

        if episode % save_interval == 0:
            ck = ckpt_dir / f"alg1_ckpt_ep{episode:05d}_{ts}.pt"
            agent.save(ck)
            tqdm.write(f"[CKPT] saved {ck.name}")

        if episode % log_interval == 0:
            recent = episode_stats[-log_interval:]
            avg_R = float(np.mean([r["sum_rate"] for r in recent]))
            avg_J = float(np.mean([r["jfi"] for r in recent]))
            avg_L = float(np.mean([r["mean_loss"] for r in recent]))
            tqdm.write(
                f"[EP {episode:5d}] train_R={avg_R:.3f}  train_JFI={avg_J:.4f}  "
                f"loss={avg_L:.3f}  eps={agent.epsilon:.3f}  "
                f"| held-out: R={eval_R:.3f}  JFI={eval_J:.4f}"
            )

    final_ck = ckpt_dir / f"alg1_ckpt_final_{ts}.pt"
    agent.save(final_ck)
    final_R, final_J = eval_greedy(env, agent, n_channels=50)

    elapsed = time.time() - t_start
    summary = {
        "timestamp": ts, "num_episodes": num_episodes,
        "total_steps": global_step, "wall_time_s": round(elapsed, 2),
        "wall_time_hr": round(elapsed / 3600.0, 4),
        "final_epsilon": agent.epsilon,
        "final_train_sum_rate": episode_stats[-1]["sum_rate"],
        "final_train_jfi": episode_stats[-1]["jfi"],
        "mean_train_sum_rate_last_50":
            float(np.mean([r["sum_rate"] for r in episode_stats[-50:]])),
        "mean_train_jfi_last_50":
            float(np.mean([r["jfi"] for r in episode_stats[-50:]])),
        "final_heldout_sum_rate": final_R,
        "final_heldout_jfi": final_J,
        "checkpoint_final": str(final_ck),
        "log_csv": str(log_path),
    }
    summary_path = run_dir / "logs" / f"train_alg1_summary_{ts}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--run-dir", type=str, default=".")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None: cfg["seed"] = args.seed
    n_ep = args.episodes if args.episodes is not None \
           else int(cfg["training"]["num_episodes"])

    env_cfg = make_env_config(cfg)
    agent_cfg = make_simple_cfg(cfg, force_cpu=args.no_cuda)
    run_dir = Path(args.run_dir).resolve()

    print("=" * 60)
    print("iRIS training -- Algorithm 1 LITERAL")
    print("=" * 60)
    print(f"[INFO] episodes     : {n_ep}")
    print(f"[INFO] lr           : {agent_cfg.learning_rate}")
    print(f"[INFO] gamma        : {agent_cfg.gamma}")
    print(f"[INFO] buffer       : NONE (Algorithm 1 uses current transition)")
    print(f"[INFO] target net   : NONE (Algorithm 1 uses online theta)")
    print()

    summary = train(cfg, env_cfg, agent_cfg, n_ep,
                    seed=int(cfg["seed"]), run_dir=run_dir)

    print(); print("=" * 60); print("TRAINING COMPLETE"); print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())