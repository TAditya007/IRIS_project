# ============================================================
# agent/__init__.py
# ------------------------------------------------------------
# Package marker for the iRIS agent modules.
# Re-exports the public API so callers can write:
#     from agent import DQNAgent, DQNConfig, ReplayBuffer, Batch
# ============================================================

from agent.replay_buffer import ReplayBuffer, Batch
from agent.dqn import DQNAgent, DQNConfig, QNetwork

__all__ = [
    "ReplayBuffer",
    "Batch",
    "DQNAgent",
    "DQNConfig",
    "QNetwork",
]