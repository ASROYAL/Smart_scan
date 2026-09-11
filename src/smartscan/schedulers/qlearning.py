"""Contextual TD-value scheduler (exposed as scheduler type ``q_learning``).

**Honest scope:** this is *not* full state-action Q-learning. It does not learn
values for (global_state, action) pairs, and it bootstraps on the best value seen
across contexts rather than the value of the actual next state, so it does **not**
perform genuine multi-step lookahead / planning. It is better described as a
tabular **contextual temporal-difference value** learner: each band is summarised
by a small discrete context (buckets of activity belief, recency and hit ratio),
and it learns the expected reward of scanning a band that looks like that context.

    V(context) <- V(context) + alpha * ( r + gamma * max_c V(c) - V(context) )

Selection is epsilon-greedy over bands by their context value. State comes only
from observation-derived BandState — never ground truth. A true DQN over the full
spectrum state (or a contextual bandit, which matches the one-step feedback well)
is the natural next step for real state-transition learning.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


def _context_key(state: BandState, current_time: float, time_scale: float) -> tuple[int, int, int]:
    """Discretise a band's state into a small context bucket."""
    # activity belief -> 4 buckets
    act = int(np.clip(state.rolling_activity_prob * 4, 0, 3))
    # recency of scan -> 3 buckets (recent / medium / long-ago-or-never)
    tss = state.time_since_scan(current_time)
    rec = 2 if np.isinf(tss) else int(np.clip(tss / max(time_scale, 1e-06) * 2, 0, 2))
    # hit ratio -> 3 buckets
    hr = int(np.clip(state.activity_ratio * 3, 0, 2))
    return (act, rec, hr)


class QLearningScheduler(BaseScheduler):
    """Tabular contextual Q-learning scheduler."""

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        scheduler_config: SchedulerConfig,
        num_bands: int,
        seed: int = 0,
    ) -> None:
        self._rx = receiver_config
        self._cfg = scheduler_config
        self._num_bands = num_bands
        self._rng = np.random.default_rng(seed)
        self._time_scale = max(scheduler_config.starvation_threshold, 1e-6)

        self._q: dict[tuple[int, int, int], float] = {}
        self._alpha = scheduler_config.q_alpha
        self._gamma = scheduler_config.q_gamma
        self._epsilon = scheduler_config.q_epsilon
        self._epsilon_decay = scheduler_config.q_epsilon_decay

        self._last_key: tuple[int, int, int] | None = None

    @property
    def name(self) -> str:
        return "q_learning"

    def _q_of(self, key: tuple[int, int, int]) -> float:
        return self._q.get(key, 0.0)

    def select_band(self, band_states: list[BandState], current_time: float) -> ScanDecision:
        keys = [_context_key(s, current_time, self._time_scale) for s in band_states]

        if self._rng.random() < self._epsilon:
            best_id = int(self._rng.integers(len(band_states)))
        else:
            values = np.array([self._q_of(k) for k in keys])
            # random tie-break among the best
            best = np.flatnonzero(values == values.max())
            best_id = int(self._rng.choice(best))

        self._epsilon *= self._epsilon_decay
        self._last_key = keys[best_id]

        s = band_states[best_id]
        return ScanDecision(
            band_id=best_id, center_frequency=s.center_frequency,
            bandwidth=min(s.bandwidth, self._rx.instantaneous_bandwidth),
            dwell_time=self._rx.dwell_time,
            priority_score=float(self._q_of(keys[best_id])),
            reason="q-learning", timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation,
               reward: float = 0.0) -> None:
        if self._last_key is None:
            return
        # TD update toward the shared reward, bootstrapping on the best known
        # context value (a contextual TD-value update — see class docstring).
        best_future = max(self._q.values()) if self._q else 0.0
        key = self._last_key
        old = self._q.get(key, 0.0)
        self._q[key] = old + self._alpha * (reward + self._gamma * best_future - old)

    def reset(self) -> None:
        self._q.clear()
        self._epsilon = self._cfg.q_epsilon
        self._last_key = None
