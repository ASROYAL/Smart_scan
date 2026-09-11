"""Non-contextual multi-armed bandit schedulers: UCB1 and Thompson Sampling.

Each frequency band is an arm. Pulling an arm = scanning that band. Reward comes
from the detection outcome (NOT ground truth). These provide a learning baseline
between the simple priority scheduler and the full contextual adaptive scheduler.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


class UCB1BanditScheduler(BaseScheduler):
    """UCB1: select arm maximizing mean_reward + c*sqrt(ln(total)/count).

    The exploration bonus guarantees every band is tried and shrinks as a band
    accumulates observations.
    """

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        scheduler_config: SchedulerConfig,
        num_bands: int,
    ) -> None:
        self._rx_config = receiver_config
        self._cfg = scheduler_config
        self._num_bands = num_bands
        self._counts = np.zeros(num_bands, dtype=np.int64)
        self._mean_reward = np.zeros(num_bands, dtype=np.float64)
        self._total = 0

    @property
    def name(self) -> str:
        return "bandit_ucb"

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        # Pull any never-tried arm first
        untried = np.where(self._counts == 0)[0]
        if len(untried) > 0:
            band_id = int(untried[0])
            score = float("inf")
            reason = "ucb explore (untried)"
        else:
            exploration = self._cfg.ucb_c * np.sqrt(
                np.log(self._total) / self._counts
            )
            ucb_values = self._mean_reward + exploration
            band_id = int(np.argmax(ucb_values))
            score = float(ucb_values[band_id])
            reason = "ucb exploit/explore"

        state = band_states[band_id]
        return ScanDecision(
            band_id=band_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=score if np.isfinite(score) else 1e9,
            reason=reason,
            timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation,
               reward: float = 0.0) -> None:
        band_id = decision.band_id
        self._counts[band_id] += 1
        self._total += 1
        # Incremental mean of the shared reward
        n = self._counts[band_id]
        self._mean_reward[band_id] += (reward - self._mean_reward[band_id]) / n

    def reset(self) -> None:
        self._counts = np.zeros(self._num_bands, dtype=np.int64)
        self._mean_reward = np.zeros(self._num_bands, dtype=np.float64)
        self._total = 0


class ThompsonSamplingScheduler(BaseScheduler):
    """Thompson Sampling with Beta-Bernoulli arms.

    Each band has a Beta(alpha, beta) posterior over its activity probability.
    Selection samples from each posterior and picks the max — naturally balancing
    exploration and exploitation.
    """

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        num_bands: int,
        seed: int = 0,
    ) -> None:
        self._rx_config = receiver_config
        self._num_bands = num_bands
        self._alpha = np.ones(num_bands, dtype=np.float64)
        self._beta = np.ones(num_bands, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def name(self) -> str:
        return "bandit_thompson"

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        samples = self._rng.beta(self._alpha, self._beta)
        band_id = int(np.argmax(samples))
        state = band_states[band_id]
        return ScanDecision(
            band_id=band_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=float(samples[band_id]),
            reason="thompson sample",
            timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation,
               reward: float = 0.0) -> None:
        # Beta-Bernoulli needs a binary outcome; treat a positive shared reward
        # (i.e. a useful, confident detection) as a success.
        band_id = decision.band_id
        if reward > 0.0:
            self._alpha[band_id] += 1
        else:
            self._beta[band_id] += 1

    def reset(self) -> None:
        self._alpha = np.ones(self._num_bands, dtype=np.float64)
        self._beta = np.ones(self._num_bands, dtype=np.float64)
        self._rng = np.random.default_rng(self._seed)
