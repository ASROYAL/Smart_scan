"""Single shared scan-reward function used by the runner AND every scheduler.

One definition of reward keeps the runner's reported `avg_reward` consistent with
what each learning scheduler actually optimises.

    reward = detection_reward
           - tuning_cost - dwell_cost
           - repeated_recent_visit_penalty      (discourages over-camping one band)
           + starvation_bonus                    (encourages returning to neglected bands)

Note the corrected direction: a band that has been *neglected for a long time* earns
a small bonus for being revisited (aiding coverage recovery), while revisiting a band
that was *just* seen is what gets penalised — the opposite of the old shaping, which
penalised long gaps and so discouraged starvation recovery.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import SchedulerConfig
from smartscan.core.models import BandObservation


def compute_scan_reward(
    cfg: SchedulerConfig,
    obs: BandObservation,
    time_since_prev_scan: float,
    starvation_threshold: float,
) -> float:
    """Compute the shaped reward for one executed scan."""
    reward = cfg.reward_hit * (0.5 + 0.5 * obs.confidence) if obs.detected else cfg.reward_miss
    reward -= cfg.reward_tuning_cost
    reward -= cfg.reward_dwell_cost

    thr = max(starvation_threshold, 1e-6)
    if not np.isinf(time_since_prev_scan):
        frac = min(time_since_prev_scan / thr, 1.0)   # 0 = just visited, 1 = long neglected
        reward -= cfg.reward_repeat_penalty_scale * (1.0 - frac)   # too-soon revisit
        reward += cfg.reward_starvation_bonus_scale * frac         # neglected-band recovery
    return reward
