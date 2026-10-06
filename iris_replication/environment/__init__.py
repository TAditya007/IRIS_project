# ============================================================
# environment/__init__.py
# ------------------------------------------------------------
# Package marker for the iRIS environment modules.
# Re-exports the public API so callers can write:
#     from environment import RicianChannel, IRISEnv, EnvConfig
# ============================================================

from environment.channel import RicianChannel
from environment.allocation import (
    SINRResult,
    compute_sinr,
    configure_phases,
    dbm_to_watt,
    jains_fairness_index,
)

__all__ = [
    "RicianChannel",
    "SINRResult",
    "compute_sinr",
    "configure_phases",
    "dbm_to_watt",
    "jains_fairness_index",
]