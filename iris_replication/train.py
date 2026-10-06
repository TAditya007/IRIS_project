# ============================================================
# train.py
# ------------------------------------------------------------
# Training script for the iRIS framework (Algorithm 1).
#
# Paper reference:
#   Algorithm 1 -- Policy Driven DNN-Based RIS Allocation Framework
#     for episode in 1..E:
#       state = env.reset()
#       for t in 0..T-1:
#         action = epsilon-greedy(Q(s, .))
#         r, s', done = env.step(action)
#         buffer.push(s, a, r, s', done)
#         if buffer.size >= train_start:
#           loss = update(buffer.sample(batch_size))
#         s = s'
#         if done: break
#       epsilon = max(eps_end, epsilon * eps_decay)
#       if step % target_update == 0: sync_target()
#       log episode metrics (sum_rate, JFI, loss, epsilon)
#
# Paper Section IV states that iRIS "achiev[es] convergence after
# approximately 3,000 episodes" but does NOT publish a loss curve.
# To honor that claim rigorously, we additionally evaluate the
# current greedy policy on a FIXED held-out set of 50 channels
# every log_interval episodes. The held-out sum-rate / JFI curves
# are the primary convergence evidence (Section IV, Fig. 6).
#
# Usage
# -----
#   python train.py                    # uses config (5000 episodes)
#   python train.py --episodes 100     # smoke run
#   python train.py --seed 123         # override seed
#   python train.py --no-cuda          # force CPU
#
# Outputs
# -------
#   logs/train_log_<timestamp>.csv         per-episode metrics (train + eval)
#   checkpoints/ckpt_ep<NNNNN>_<ts>.pt     periodic checkpoints
#   logs/train_summary_<ts>.txt            final summary
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
from agent.replay_buffer import ReplayBuffer
from agent.dqn import DQNAgent, DQNConfig


# ============================================================
# Config loading
# ============================================================
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config (Rule R4)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_config(cfg: Dict[str, Any]) -> EnvConfig:
    """Map YAML config to EnvConfig dataclass."""
    sys_cfg = cfg["system"]
    ch_cfg = cfg["channel"]
    env_cfg = cfg["env"]
    rew_cfg = cfg["reward"]

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
        seed=int(cfg["seed"]),
    )


def make_dqn_config(cfg: Dict[str, Any], force_cpu: bool) -> DQNConfig:
    """Map YAML config to DQNConfig dataclass."""
    a = cfg["agent"]
    device = "cpu" if force_cpu else cfg["training"]["device"]
    return DQNConfig(
        hidden_dim=int(a["hidden_dim"]),
        num_hidden_layers=int(a["num_hidden_layers"]),
        learning_rate=float(a["learning_rate"]),
        gamma=float(a["gamma"]),
        epsilon_start=float(a["epsilon_start"]),
        epsilon_end=float(a["epsilon_end"]),
        epsilon_decay=float(a["epsilon_decay"]),
        batch_size=int(a["batch_size"]),
        buffer_size=int(a["buffer_size"]),
        train_start=int(a["train_start"]),
        target_update=int(a["target_update"]),
        device=device,
    )


# ============================================================
# Held-out evaluation set
# ============================================================
def build_eval_channels(
    env: IRISEnv, n_channels: int = 50, seed: int = 999,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Draw a fixed set of (h, g) channel blocks for held-out evaluation.

    These channels are NOT used during training. They are used only
    to evaluate the current greedy policy, mirroring the paper's
    convergence claim (Section IV: "convergence after approximately
    3,000 episodes").
    """
    rng = np.random.default_rng(seed)
    channels: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_channels):
        h, g = env.channel.sample_block()
        channels.append((h, g))
    return channels


def eval_policy_greedy(
    env: IRISEnv,
    agent: DQNAgent,
    channels: List[Tuple[np.ndarray, np.ndarray]],
    eval_seed: int = 12345,
) -> Tuple[float, float]:
    """
    Evaluate the current greedy policy on a fixed held-out set of
    channels.

    Uses its own dedicated RNG and restores training state afterwards
    so that eval metrics are directly comparable across episodes and
    do not perturb training.
    """
    # --- Save training env state ---
    saved_rng = env.rng
    saved_h = env.h.copy()
    saved_g = env.g.copy()
    saved_alloc = {k: list(v) for k, v in env.allocation.items()}
    saved_next = env.next_element_idx
    saved_t = env.t

    # --- Fresh RNG for eval (deterministic across calls) ---
    env.rng = np.random.default_rng(eval_seed)

    sum_rates: List[float] = []
    jfis: List[float] = []
    N = env.cfg.N

    try:
        for h, g in channels:
            env.h = h
            env.g = g
            env.allocation = {i: [] for i in range(N)}
            env.next_element_idx = 0
            env.t = 0

            state = env._get_state()
            done = False
            info = None
            while not done:
                a = agent.select_action(state, greedy=True)
                state, _r, done, info = env.step(a)

            sum_rates.append(float(info["sum_rate"]))
            jfis.append(float(info["jfi"]))
    finally:
        # --- Restore training env state ---
        env.rng = saved_rng
        env.h = saved_h
        env.g = saved_g
        env.allocation = saved_alloc
        env.next_element_idx = saved_next
        env.t = saved_t

    return float(np.mean(sum_rates)), float(np.mean(jfis))


# ============================================================
# CSV logger
# ============================================================
class TrainLogger:
    """Append-only CSV of per-episode metrics (train + held-out eval)."""

    FIELDS = [
        "episode",
        "steps",
        "sum_rate",
        "mean_rate",
        "jfi",
        "total_reward",
        "mean_reward",
        "mean_loss",
        "epsilon",
        "n_updates",
        "wall_time_s",
        "eval_sum_rate",   # held-out sum-rate (NaN unless episode % log_interval == 0)
        "eval_jfi",        # held-out JFI
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writerow(row)


# ============================================================
# Main training loop (Algorithm 1)
# ============================================================
def train(
    cfg: Dict[str, Any],
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    num_episodes: int,
    seed: int,
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Execute Algorithm 1. Returns final summary dict.
    """
    # --- Set all seeds (Rule R5) ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Build env, buffer, agent ---
    env = IRISEnv(env_cfg)
    buffer = ReplayBuffer(capacity=dqn_cfg.buffer_size,
                          state_dim=env.state_dim, seed=seed)
    agent = DQNAgent(state_dim=env.state_dim,
                     action_dim=env.action_dim,
                     cfg=dqn_cfg, seed=seed)

    # --- Held-out eval set (paper: Section IV convergence claim) ---
    eval_channels = build_eval_channels(env, n_channels=50, seed=999)

    # --- Prepare run artifacts ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / "logs" / f"train_log_{ts}.csv"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(log_path)

    save_interval = int(cfg["training"]["save_interval"])
    log_interval = int(cfg["training"]["log_interval"])

    print(f"[INFO] env: M={env_cfg.M}, N={env_cfg.N}, "
          f"state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"[INFO] agent on device: {agent.device}")
    print(f"[INFO] training for {num_episodes} episodes")
    print(f"[INFO] held-out eval: {len(eval_channels)} channels, "
          f"evaluated every {log_interval} episodes")
    print(f"[INFO] log: {log_path}")
    print(f"[INFO] checkpoints: {ckpt_dir}")
    print()

    # --- Training loop ---
    t_start = time.time()
    global_step = 0
    episode_stats: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, float]] = []

    pbar = tqdm(range(1, num_episodes + 1),
                desc="train", unit="ep", ncols=110)

    for episode in pbar:
        state = env.reset()
        ep_reward = 0.0
        ep_losses: List[float] = []
        ep_info = None
        steps = 0
        done = False

        while not done:
            # --- Algorithm 1: action selection ---
            action = agent.select_action(state, greedy=False)

            # --- Step env ---
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_info = info
            steps += 1

            # --- Store transition ---
            buffer.push(state, action, reward, next_state, done)

            # --- Learn (if enough samples) ---
            if buffer.is_ready(dqn_cfg.batch_size):
                batch = buffer.sample(dqn_cfg.batch_size)
                loss = agent.update(batch)
                ep_losses.append(loss)

            state = next_state
            global_step += 1

        # --- End of episode: epsilon decay (Algorithm 1) ---
        agent.decay_epsilon()

        # --- Held-out evaluation every log_interval episodes ---
        eval_R = float("nan")
        eval_J = float("nan")
        if episode % log_interval == 0:
            eval_R, eval_J = eval_policy_greedy(env, agent, eval_channels)
            eval_history.append({"episode": episode,
                                 "eval_sum_rate": eval_R,
                                 "eval_jfi": eval_J})

        # --- Episode metrics ---
        sum_rate = float(ep_info["sum_rate"]) if ep_info else 0.0
        rate_vec = ep_info["rate"] if ep_info else np.zeros(env_cfg.N)
        jfi = float(ep_info["jfi"]) if ep_info else 0.0
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        row = {
            "episode": episode,
            "steps": steps,
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
        logger.log(row)
        episode_stats.append(row)

        # --- Progress bar ---
        pbar.set_postfix({
            "R": f"{sum_rate:.2f}",
            "JFI": f"{jfi:.3f}",
            "eps": f"{agent.epsilon:.3f}",
            "loss": f"{mean_loss:.3f}",
        })

        # --- Periodic checkpoint ---
        if episode % save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_ep{episode:05d}_{ts}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[CKPT] saved {ckpt_path.name}")

        # --- Periodic summary print ---
        if episode % log_interval == 0:
            recent = episode_stats[-log_interval:]
            avg_R = float(np.mean([r["sum_rate"] for r in recent]))
            avg_J = float(np.mean([r["jfi"] for r in recent]))
            avg_L = float(np.mean([r["mean_loss"] for r in recent]))
            tqdm.write(
                f"[EP {episode:5d}] "
                f"train_R={avg_R:.3f}  train_JFI={avg_J:.4f}  "
                f"loss={avg_L:.4f}  eps={agent.epsilon:.3f}  "
                f"| held-out: R={eval_R:.3f}  JFI={eval_J:.4f}"
            )

    # --- Final checkpoint ---
    final_ckpt = ckpt_dir / f"ckpt_final_{ts}.pt"
    agent.save(final_ckpt)

    # --- Final held-out evaluation ---
    final_eval_R, final_eval_J = eval_policy_greedy(
        env, agent, eval_channels
    )

    # --- Summary ---
    elapsed = time.time() - t_start
    summary = {
        "timestamp": ts,
        "num_episodes": num_episodes,
        "total_steps": global_step,
        "wall_time_s": round(elapsed, 2),
        "wall_time_hr": round(elapsed / 3600.0, 4),
        "final_epsilon": agent.epsilon,
        "final_train_sum_rate": episode_stats[-1]["sum_rate"],
        "final_train_jfi": episode_stats[-1]["jfi"],
        "mean_train_sum_rate_last_50": float(np.mean(
            [r["sum_rate"] for r in episode_stats[-50:]])),
        "mean_train_jfi_last_50": float(np.mean(
            [r["jfi"] for r in episode_stats[-50:]])),
        "final_heldout_sum_rate": final_eval_R,
        "final_heldout_jfi": final_eval_J,
        "checkpoint_final": str(final_ckpt),
        "log_csv": str(log_path),
    }
    summary_path = run_dir / "logs" / f"train_summary_{ts}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return summary


# ============================================================
# Entry point
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train iRIS (Algorithm 1)"
    )
    parser.add_argument("--config", type=str,
                        default="configs/default.yaml",
                        help="path to YAML config")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override num_episodes (smoke runs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override seed")
    parser.add_argument("--no-cuda", action="store_true",
                        help="force CPU")
    parser.add_argument("--run-dir", type=str, default=".",
                        help="output root dir")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    num_episodes = (args.episodes
                    if args.episodes is not None
                    else int(cfg["training"]["num_episodes"]))

    env_cfg = make_env_config(cfg)
    dqn_cfg = make_dqn_config(cfg, force_cpu=args.no_cuda)

    run_dir = Path(args.run_dir).resolve()
    print("=" * 60)
    print("iRIS training (Algorithm 1)")
    print("=" * 60)
    print(f"[INFO] config       : {args.config}")
    print(f"[INFO] seed         : {cfg['seed']}")
    print(f"[INFO] episodes     : {num_episodes}")
    print(f"[INFO] batch_size   : {dqn_cfg.batch_size}")
    print(f"[INFO] buffer_size  : {dqn_cfg.buffer_size}")
    print(f"[INFO] lr           : {dqn_cfg.learning_rate}")
    print(f"[INFO] gamma        : {dqn_cfg.gamma}")
    print(f"[INFO] target_update: {dqn_cfg.target_update}")
    print(f"[INFO] run_dir      : {run_dir}")
    print()

    summary = train(cfg, env_cfg, dqn_cfg, num_episodes,
                    seed=int(cfg["seed"]), run_dir=run_dir)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())